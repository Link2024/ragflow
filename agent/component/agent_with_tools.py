#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import partial
from typing import Any

import json_repair
from timeit import default_timer as timer
from agent.tools.base import LLMToolPluginCallSession, ToolParamBase, ToolBase, ToolMeta
from api.db.services.llm_service import LLMBundle
from api.db.services.tenant_llm_service import TenantLLMService
from api.db.services.mcp_server_service import MCPServerService
from api.utils.api_utils import timeout
from rag.prompts.generator import next_step, COMPLETE_TASK, analyze_task, \
    citation_prompt, reflect, rank_memories, kb_prompt, citation_plus, full_question, message_fit_in
from rag.utils.mcp_tool_call_conn import MCPToolCallSession, mcp_tool_metadata_to_openai_tool
from agent.component.llm import LLMParam, LLM
import json


class AgentParam(LLMParam, ToolParamBase):
    """
    Define the Agent component parameters.
    """

    def __init__(self):
        self.meta:ToolMeta = {
                "name": "agent",
                "description": "This is an agent for a specific task.",
                "parameters": {
                    "user_prompt": {
                        "type": "string",
                        "description": (
                            "The task instruction for the agent. "
                            "Provide a clear, concise description of what the agent needs to do. "
                            "If the task requires exact user input, use 'original_input' parameter instead."
                        ),
                        "default": "",
                        "required": True
                    },
                    "reasoning": {
                        "type": "string",
                        "description": (
                            "Supervisor's reasoning for choosing the this agent. "
                            "Explain why this agent is being invoked and what is expected of it."
                        ),
                        "required": True
                    },
                    "context": {
                        "type": "string",
                        "description": (
                                "All relevant background information, prior facts, decisions, "
                                "and state needed by the agent to solve the current query. "
                                "Should be as detailed and self-contained as possible."
                            ),
                        "required": True
                    },
                    "original_input": {
                        "type": "string",
                        "description": (
                            "The ORIGINAL user input when the sub-agent needs exact content. "
                            "Include this when the task involves: "
                            "(1) Text analysis, processing, or generation that requires exact wording; "
                            "(2) Data extraction with precise requirements; "
                            "(3) Content modification based on specific user input; "
                            "(4) Tasks where user's exact phrasing, terminology, or data points matter. "
                            "Omit this if the agent only needs summarized context."
                        ),
                        "required": False
                    },
                }
            }
        super().__init__()
        self.function_name = "agent"
        self.tools = []
        self.mcp = []
        self.max_rounds = 5
        self.description = ""


class Agent(LLM, ToolBase):
    component_name = "Agent"

    def __init__(self, canvas, id, param: LLMParam):
        LLM.__init__(self, canvas, id, param)
        self.tools = {}
        for cpn in self._param.tools:
            cpn = self._load_tool_obj(cpn)
            self.tools[cpn.get_meta()["function"]["name"]] = cpn

        self.chat_mdl = LLMBundle(self._canvas.get_tenant_id(), TenantLLMService.llm_id2llm_type(self._param.llm_id), self._param.llm_id,
                                  max_retries=self._param.max_retries,
                                  retry_interval=self._param.delay_after_error,
                                  max_rounds=self._param.max_rounds,
                                  verbose_tool_use=True
                                  )
        self.tool_meta = [v.get_meta() for _,v in self.tools.items()]

        for mcp in self._param.mcp:
            _, mcp_server = MCPServerService.get_by_id(mcp["mcp_id"])
            tool_call_session = MCPToolCallSession(mcp_server, mcp_server.variables)
            for tnm, meta in mcp["tools"].items():
                self.tool_meta.append(mcp_tool_metadata_to_openai_tool(meta))
                self.tools[tnm] = tool_call_session
        self.callback = partial(self._canvas.tool_use_callback, id)
        self.toolcall_session = LLMToolPluginCallSession(self.tools, self.callback)
        #self.chat_mdl.bind_tools(self.toolcall_session, self.tool_metas)

    def _load_tool_obj(self, cpn: dict) -> object:
        from agent.component import component_class
        param = component_class(cpn["component_name"] + "Param")()
        param.update(cpn["params"])
        try:
            param.check()
        except Exception as e:
            self.set_output("_ERROR", cpn["component_name"] + f" configuration error: {e}")
            raise
        cpn_id = f"{self._id}-->" + cpn.get("name", "").replace(" ", "_")
        return component_class(cpn["component_name"])(self._canvas, cpn_id, param)

    def get_meta(self) -> dict[str, Any]:
        self._param.function_name= self._id.split("-->")[-1]
        m = super().get_meta()
        if hasattr(self._param, "user_prompt") and self._param.user_prompt:
            m["function"]["parameters"]["properties"]["user_prompt"] = self._param.user_prompt
        return m

    def get_input_form(self) -> dict[str, dict]:
        res = {}
        for k, v in self.get_input_elements().items():
            res[k] = {
                "type": "line",
                "name": v["name"]
            }
        for cpn in self._param.tools:
            if not isinstance(cpn, LLM):
                continue
            res.update(cpn.get_input_form())
        return res

    @timeout(int(os.environ.get("COMPONENT_EXEC_TIMEOUT", 20*60)))
    def _invoke(self, **kwargs):
        if kwargs.get("user_prompt"):
            usr_pmt = ""
            if kwargs.get("reasoning"):
                usr_pmt += "\nREASONING:\n{}\n".format(kwargs["reasoning"])
            if kwargs.get("context"):
                usr_pmt += "\nCONTEXT:\n{}\n".format(kwargs["context"])
            if kwargs.get("original_input"):
                usr_pmt += "\nORIGINAL USER INPUT:\n{}\n".format(kwargs["original_input"])
            if usr_pmt:
                usr_pmt += "\nQUERY:\n{}\n".format(str(kwargs["user_prompt"]))
            else:
                usr_pmt = str(kwargs["user_prompt"])
            self._param.prompts = [{"role": "user", "content": usr_pmt}]

        if not self.tools:
            return LLM._invoke(self, **kwargs)

        prompt, msg, user_defined_prompt = self._prepare_prompt_variables()

        downstreams = self._canvas.get_component(self._id)["downstream"] if self._canvas.get_component(self._id) else []
        ex = self.exception_handler()
        if any([self._canvas.get_component_obj(cid).component_name.lower()=="message" for cid in downstreams]) and not self._param.output_structure and not (ex and ex["goto"]):
            self.set_output("content", partial(self.stream_output_with_tools, prompt, msg, user_defined_prompt))
            return

        _, msg = message_fit_in([{"role": "system", "content": prompt}, *msg], int(self.chat_mdl.max_length * 0.97))
        use_tools = []
        ans = ""
        for delta_ans, tk in self._react_with_tools_streamly(prompt, msg, use_tools, user_defined_prompt):
            ans += delta_ans

        if ans.find("**ERROR**") >= 0:
            logging.error(f"Agent._chat got error. response: {ans}")
            if self.get_exception_default_value():
                self.set_output("content", self.get_exception_default_value())
            else:
                self.set_output("_ERROR", ans)
            return

        self.set_output("content", ans)
        if use_tools:
            self.set_output("use_tools", use_tools)
        return ans

    def stream_output_with_tools(self, prompt, msg, user_defined_prompt={}):
        _, msg = message_fit_in([{"role": "system", "content": prompt}, *msg], int(self.chat_mdl.max_length * 0.97))
        answer_without_toolcall = ""
        use_tools = []
        for delta_ans,_ in self._react_with_tools_streamly(prompt, msg, use_tools, user_defined_prompt):
            if delta_ans.find("**ERROR**") >= 0:
                if self.get_exception_default_value():
                    self.set_output("content", self.get_exception_default_value())
                    yield self.get_exception_default_value()
                else:
                    self.set_output("_ERROR", delta_ans)
            answer_without_toolcall += delta_ans
            yield delta_ans

        self.set_output("content", answer_without_toolcall)
        if use_tools:
            self.set_output("use_tools", use_tools)

    def _gen_citations(self, text):
        retrievals = self._canvas.get_reference()
        retrievals = {"chunks": list(retrievals["chunks"].values()), "doc_aggs": list(retrievals["doc_aggs"].values())}
        formated_refer = kb_prompt(retrievals, self.chat_mdl.max_length, True)
        for delta_ans in self._generate_streamly([{"role": "system", "content": citation_plus("\n\n".join(formated_refer))},
                                                  {"role": "user", "content": text}
                                                  ]):
            yield delta_ans

    def _react_with_tools_streamly(self, prompt, history: list[dict], use_tools, user_defined_prompt={}):
        token_count = 0
        tool_metas = self.tool_meta
        hist = deepcopy(history)
        last_calling = ""
        if len(hist) > 3:
            st = timer()
            user_request = full_question(messages=history, chat_mdl=self.chat_mdl)
            self.callback("Multi-turn conversation optimization", {}, user_request, elapsed_time=timer()-st)
        else:
            user_request = history[-1]["content"]

        def use_tool(name, args):
            nonlocal hist, use_tools, token_count, last_calling, user_request
            import json
            import re
            
            # === 输入参数日志 ===
            logging.info(f"\n{'='*80}")
            logging.info(f"🔧 [子Agent调用] {name}")
            logging.info(f"   父Agent: {self._id}")
            try:
                full_args = json.dumps(args, ensure_ascii=False, indent=2)
                logging.info(f"   完整参数:\n{full_args}")
            except Exception as e:
                logging.error(f"   参数序列化失败: {e}")
                logging.info(f"   原始参数: {args}")
            logging.info(f"{'='*80}\n")

            last_calling = name
            
            # 1. 执行调用
            tool_response = self.toolcall_session.tool_call(name, args)
            
            # === 原始返回结果 ===
            logging.info(f"\n{'='*80}")
            logging.info(f"📦 [子Agent原始返回] {name}")
            logging.info(f"   类型: {type(tool_response)}")
            if tool_response is None:
                logging.warning(f"   ⚠️  返回为 None")
            else:
                try:
                    if isinstance(tool_response, (dict, list)):
                        raw_json = json.dumps(tool_response, ensure_ascii=False, indent=2)
                        logging.info(f"   内容:\n{raw_json}")
                    else:
                        logging.info(f"   内容:\n{str(tool_response)}")
                except Exception as e:
                    logging.error(f"   序列化失败: {e}")
                    logging.info(f"   repr: {repr(tool_response)}")
            logging.info(f"{'='*80}\n")

            # ================= 核心修复逻辑开始 =================
            
            # 2. 【补救】如果直接返回是 None，尝试从工具对象的状态中获取 output
            # (这是解决你日志中“返回值为 NULL”但“完整输出”里有数据的关键)
            if tool_response is None:
                tool_obj = self.tools.get(name)
                if tool_obj and hasattr(tool_obj, 'output'):
                    rescued_data = tool_obj.output()
                    if rescued_data:
                        tool_response = rescued_data
                        logging.info(f" 🔧 [自动修复] 已从 Tool.output() 补救回数据")

            # 3. 【清洗】提取 content 并转为字符串
            actual_response = tool_response # 默认为原值
            
            if isinstance(tool_response, dict):
                # 情况A: 标准字典返回 {'content': '...', ...}
                if 'content' in tool_response:
                    actual_response = tool_response['content']
                    logging.info(f" 🧹 [数据清洗] 提取 content 字段成功")
                else:
                    # 情况B: 字典但没有content，转字符串防止丢数据
                    actual_response = json.dumps(tool_response, ensure_ascii=False)
            
            # 4. 【去噪】去除 Markdown 包裹 (兼容修正Agent返回的纯代码)
            # 匹配 ```json ... ``` 或 ```python ... ``` 或 纯 ``` ... ```
            if isinstance(actual_response, str):
                pattern = r"```(?:\w+)?\s*(.*?)```"
                match = re.search(pattern, actual_response, re.DOTALL)
                if match:
                    actual_response = match.group(1).strip()
                    logging.info(f" 🧹 [数据清洗] 去除 Markdown 代码块包裹成功")

            # ================= 核心修复逻辑结束 =================

            # === 清洗后结果 ===
            logging.info(f"\n{'='*80}")
            logging.info(f"✅ [清洗后最终结果] {name}")
            if actual_response is None:
                logging.error(f"   ⚠️  清洗后仍为 None")
                tool_obj = self.tools.get(name)
                if tool_obj and hasattr(tool_obj, 'output'):
                    logging.error(f"   子Agent对象状态: {tool_obj.output()}")
            else:
                logging.info(f"   类型: {type(actual_response)}")
                logging.info(f"   内容:\n{str(actual_response)}")
            logging.info(f"{'='*80}\n")
            
            # 5. 存入历史
            # ⚠️ 关键修正：这里必须存 actual_response (清洗后的字符串)，
            # 绝对不能存 tool_response (可能是 None 或 复杂字典)
            use_tools.append({
                "name": name,
                "arguments": args,
                "results": actual_response 
            })
            
            # self.callback("add_memory", {}, "...")
            # self.add_memory(hist[-2]["content"], hist[-1]["content"], name, args, str(actual_response), user_defined_prompt)

            return name, actual_response

        def complete():
            nonlocal hist
            need2cite = self._param.cite and self._canvas.get_reference()["chunks"] and self._id.find("-->") < 0
            cited = False
            if hist[0]["role"] == "system" and need2cite:
                if len(hist) < 7:
                    hist[0]["content"] += citation_prompt()
                    cited = True
            yield "", token_count

            _hist = hist
            if len(hist) > 12:
                _hist = [hist[0], hist[1], *hist[-10:]]
            entire_txt = ""
            for delta_ans in self._generate_streamly(_hist):
                if not need2cite or cited:
                    yield delta_ans, 0
                entire_txt += delta_ans
            if not need2cite or cited:
                return

            st = timer()
            txt = ""
            for delta_ans in self._gen_citations(entire_txt):
                yield delta_ans, 0
                txt += delta_ans

            self.callback("gen_citations", {}, txt, elapsed_time=timer()-st)

        def append_user_content(hist, content):
            if hist[-1]["role"] == "user":
                hist[-1]["content"] += content
            else:
                hist.append({"role": "user", "content": content})

        st = timer()
        task_desc = analyze_task(self.chat_mdl, prompt, user_request, tool_metas, user_defined_prompt)
        self.callback("analyze_task", {}, task_desc, elapsed_time=timer()-st)
        for _ in range(self._param.max_rounds + 1):
            response, tk = next_step(self.chat_mdl, hist, tool_metas, task_desc, user_defined_prompt)
            # self.callback("next_step", {}, str(response)[:256]+"...")
            token_count += tk
            hist.append({"role": "assistant", "content": response})
            
            # 自动注入 original_input（如果 TASK ANALYSIS 中有且工具需要）
            try:
                match = re.search(r'"Original User Input":\s*"((?:[^"\\]|\\.)*)"', task_desc, re.DOTALL)
                if match:
                    original_input = match.group(1).encode('utf-8').decode('unicode_escape')
                    functions = json_repair.loads(re.sub(r"```.*", "", response))
                    if isinstance(functions, list):
                        modified = False
                        for func in functions:
                            if isinstance(func, dict) and 'arguments' in func:
                                tool_meta = next((t for t in tool_metas if t['function']['name'] == func['name']), None)
                                if tool_meta:
                                    params = tool_meta['function']['parameters']['properties']
                                    if 'original_input' in params and 'original_input' not in func['arguments']:
                                        func['arguments']['original_input'] = original_input
                                        modified = True
                                        logging.info(f"🔧 [自动注入] {func['name']} <- original_input")
                        if modified:
                            response = json.dumps(functions, ensure_ascii=False)
                            hist[-1]["content"] = response
            except Exception as e:
                logging.debug(f"original_input 自动注入跳过: {e}")
            
            try:
                functions = json_repair.loads(re.sub(r"```.*", "", response))
                if not isinstance(functions, list):
                    raise TypeError(f"List should be returned, but `{functions}`")
                for f in functions:
                    if not isinstance(f, dict):
                        raise TypeError(f"An object type should be returned, but `{f}`")
                with ThreadPoolExecutor(max_workers=5) as executor:
                    thr = []
                    for func in functions:
                        name = func["name"]
                        args = func["arguments"]
                        if name == COMPLETE_TASK:
                            append_user_content(hist, f"Respond with a formal answer. FORGET(DO NOT mention) about `{COMPLETE_TASK}`. The language for the response MUST be as the same as the first user request.\n")
                            for txt, tkcnt in complete():
                                yield txt, tkcnt
                            return

                        thr.append(executor.submit(use_tool, name, args))

                    st = timer()
                    reflection = reflect(self.chat_mdl, hist, [th.result() for th in thr], user_defined_prompt)
                    append_user_content(hist, reflection)
                    self.callback("reflection", {}, str(reflection), elapsed_time=timer()-st)

            except Exception as e:
                logging.exception(msg=f"Wrong JSON argument format in LLM ReAct response: {e}")
                e = f"\nTool call error, please correct the input parameter of response format and call it again.\n *** Exception ***\n{e}"
                append_user_content(hist, str(e))

        logging.warning( f"Exceed max rounds: {self._param.max_rounds}")
        final_instruction = f"""
{user_request}
IMPORTANT: You have reached the conversation limit. Based on ALL the information and research you have gathered so far, please provide a DIRECT and COMPREHENSIVE final answer to the original request.
Instructions:
1. SYNTHESIZE all information collected during this conversation
2. Provide a COMPLETE response using existing data - do not suggest additional research
3. Structure your response as a FINAL DELIVERABLE, not a plan
4. If information is incomplete, state what you found and provide the best analysis possible with available data
5. DO NOT mention conversation limits or suggest further steps
6. Focus on delivering VALUE with the information already gathered
Respond immediately with your final comprehensive answer.
        """
        append_user_content(hist, final_instruction)

        for txt, tkcnt in complete():
            yield txt, tkcnt

    def get_useful_memory(self, goal: str, sub_goal:str, topn=3, user_defined_prompt:dict={}) -> str:
        # self.callback("get_useful_memory", {"topn": 3}, "...")
        mems = self._canvas.get_memory()
        rank = rank_memories(self.chat_mdl, goal, sub_goal, [summ for (user, assist, summ) in mems], user_defined_prompt)
        try:
            rank = json_repair.loads(re.sub(r"```.*", "", rank))[:topn]
            mems = [mems[r] for r in rank]
            return "\n\n".join([f"User: {u}\nAgent: {a}" for u, a,_ in mems])
        except Exception as e:
            logging.exception(e)

        return "Error occurred."

    def reset(self, temp=False):
        """
        Reset all tools if they have a reset method. This avoids errors for tools like MCPToolCallSession.
        """
        for k, cpn in self.tools.items():
            if hasattr(cpn, "reset") and callable(cpn.reset):
                cpn.reset()

