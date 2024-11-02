# encoding:utf-8

import time

import openai
import openai.error

from bot.bot import Bot
from bot.deepseek.deepseek_session import DeepSeekSession
from bot.session_manager import SessionManager
from bridge.context import ContextType
from bridge.reply import Reply, ReplyType
from common.log import logger
from config import conf

user_session = dict()

#模型(1)	       上下文长度	最大输出长度(2)	
#deepseek-chat	  128K	       4K (8KBeta)	
# OpenAI对话模型API (可用)
class DeepSeekBot(Bot):
    def __init__(self):
        super().__init__()
        openai.api_key = conf().get("deepseek_api_key")
        if conf().get("deepseek_api_api_base"):
            openai.api_base = conf().get("deepseek_api_api_base")

        self.sessions = SessionManager(DeepSeekSession, model=conf().get("model") or "deepseek-chat")
        self.args = {
            "model": conf().get("model") or "deepseek-chat",  # 对话模型的名称
            # temperature
            #代码生成/数学解题   	0.0
            #数据抽取/分析	1.0
            #通用对话	1.3
            #翻译	1.3
            #创意类写作/诗歌创作	1.5
            "temperature": conf().get("temperature", 1.3),  # 值在[0,1]之间，越大表示回复越具有不确定性
            "max_tokens": 4096,  # 回复最大的字符数
            "request_timeout": conf().get("request_timeout", None),  # 请求超时时间，openai接口默认设置为600，对于难问题一般需要较长时间
            "timeout": conf().get("request_timeout", None),  # 重试超时时间，在这个时间内，将会自动重试
            "stop": ["\n\n\n"],
        }

    def reply(self, query, context=None):
        # acquire reply content
        if context and context.type:
            if context.type == ContextType.TEXT:
                logger.info("[DEEPSEEK_AI] query={}".format(query))
                session_id = context["session_id"]
                reply = None
                if query == "#清除记忆":
                    self.sessions.clear_session(session_id)
                    reply = Reply(ReplyType.INFO, "记忆已清除")
                elif query == "#清除所有":
                    self.sessions.clear_all_session()
                    reply = Reply(ReplyType.INFO, "所有人记忆已清除")
                else:
                    session = self.sessions.session_query(query, session_id)
                    result = self.reply_text(session)
                    total_tokens, completion_tokens, reply_content = (
                        result["total_tokens"],
                        result["completion_tokens"],
                        result["content"],
                    )
                    logger.debug(
                        "[DEEPSEEK_AI] new_query={}, session_id={}, reply_cont={}, completion_tokens={}".format(str(session), session_id, reply_content, completion_tokens)
                    )

                    if total_tokens == 0:
                        reply = Reply(ReplyType.ERROR, reply_content)
                    else:
                        self.sessions.session_reply(reply_content, session_id, total_tokens)
                        reply = Reply(ReplyType.TEXT, reply_content)
                return reply
            elif context.type == ContextType.IMAGE_CREATE:
                ok, retstring = self.create_img(query, 0)
                reply = None
                if ok:
                    reply = Reply(ReplyType.IMAGE_URL, retstring)
                else:
                    reply = Reply(ReplyType.ERROR, retstring)
                return reply

    def reply_text(self, session: DeepSeekSession, retry_count=0):
        try:
            response = openai.Completion.create(prompt=str(session), **self.args)
            res_content = response.choices[0]["text"].strip().replace("<|endoftext|>", "")
            total_tokens = response["usage"]["total_tokens"]
            completion_tokens = response["usage"]["completion_tokens"]
            logger.info("[DEEPSEEK_AI] reply={}".format(res_content))
            return {
                "total_tokens": total_tokens,
                "completion_tokens": completion_tokens,
                "content": res_content,
            }
        except Exception as e:
            need_retry = retry_count < 2
            result = {"completion_tokens": 0, "content": "我现在有点累了，等会再来吧"}
            if isinstance(e, openai.error.RateLimitError):
                logger.warn("[DEEPSEEK_AI] RateLimitError: {}".format(e))
                result["content"] = "提问太快啦，请休息一下再问我吧"
                if need_retry:
                    time.sleep(20)
            elif isinstance(e, openai.error.Timeout):
                logger.warn("[DEEPSEEK_AI] Timeout: {}".format(e))
                result["content"] = "我没有收到你的消息"
                if need_retry:
                    time.sleep(5)
            elif isinstance(e, openai.error.APIConnectionError):
                logger.warn("[DEEPSEEK_AI] APIConnectionError: {}".format(e))
                need_retry = False
                result["content"] = "我连接不到你的网络"
            else:
                logger.warn("[DEEPSEEK_AI] Exception: {}".format(e))
                need_retry = False
                self.sessions.clear_session(session.session_id)

            if need_retry:
                logger.warn("[DEEPSEEK_AI] 第{}次重试".format(retry_count + 1))
                return self.reply_text(session, retry_count + 1)
            else:
                return result
