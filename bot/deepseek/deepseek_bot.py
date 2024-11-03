# encoding:utf-8

import time

import openai
from openai import OpenAI
from bot.deepseek.deepseek_session import DeepSeekSession
from bot.bot import Bot
from bot.chatgpt.chat_gpt_session import ChatGPTSession
from bot.session_manager import SessionManager
from bridge.context import ContextType
from bridge.reply import Reply, ReplyType
from common.log import logger
from config import conf, load_config

# OpenAI对话模型API (可用)
class DeepSeekBot(Bot):
    def __init__(self):
        super().__init__()
        self.client = OpenAI(api_key=conf().get("deepseek_api_key"), base_url="https://api.deepseek.com")
        self.sessions = SessionManager(DeepSeekSession, model="deepseek-chat")
        # o1相关模型不支持system prompt，暂时用文心模型的session

        self.args = {
            "model": "deepseek-chat",  # 对话模型的名称
            # temperature
            #代码生成/数学解题   	0.0
            #数据抽取/分析	1.0
            #通用对话	1.3
            #翻译	1.3
            #创意类写作/诗歌创作	1.5
            "temperature": conf().get("temperature", 1.3),  # 值在[0,1]之间，越大表示回复越具有不确定性
            "max_tokens": 4096,  # 回复最大的字符数
            "top_p": 1,
            "frequency_penalty": conf().get("frequency_penalty", 0.0),  # [-2,2]之间，该值越大则更倾向于产生不同的内容
            "presence_penalty": conf().get("presence_penalty", 0.0),  # [-2,2]之间，该值越大则更倾向于产生不同的内容
            #"request_timeout": conf().get("request_timeout", None),  # 请求超时时间，openai接口默认设置为600，对于难问题一般需要较长时间
            "timeout": conf().get("request_timeout", None),  # 重试超时时间，在这个时间内，将会自动重试
        }

    def reply(self, query, context=None):
        # acquire reply content
        if context.type == ContextType.TEXT:
            logger.info("[DEEPSEEK_AI] query={}".format(query))

            session_id = context["session_id"]
            reply = None
            clear_memory_commands = conf().get("clear_memory_commands", ["#清除记忆"])
            if query in clear_memory_commands:
                self.sessions.clear_session(session_id)
                reply = Reply(ReplyType.INFO, "记忆已清除")
            elif query == "#清除所有":
                self.sessions.clear_all_session()
                reply = Reply(ReplyType.INFO, "所有人记忆已清除")
            elif query == "#更新配置":
                load_config()
                reply = Reply(ReplyType.INFO, "配置已更新")
            if reply:
                return reply
            session = self.sessions.session_query(query, session_id)
            logger.debug("[DEEPSEEK_AI] session query={}".format(session.messages))

            api_key = context.get("openai_api_key")
            reply_content = self.reply_text(session, api_key, args=self.args)
            logger.debug(
                "[DEEPSEEK_AI] new_query={}, session_id={}, reply_cont={}, completion_tokens={}".format(
                    session.messages,
                    session_id,
                    reply_content["content"],
                    reply_content["completion_tokens"],
                )
            )
            if reply_content["completion_tokens"] == 0 and len(reply_content["content"]) > 0:
                reply = Reply(ReplyType.ERROR, reply_content["content"])
            elif reply_content["completion_tokens"] > 0:
                self.sessions.session_reply(reply_content["content"], session_id, reply_content["total_tokens"])
                reply = Reply(ReplyType.TEXT, reply_content["content"])
            else:
                reply = Reply(ReplyType.ERROR, reply_content["content"])
                logger.debug("[DEEPSEEK_AI] reply {} used 0 tokens.".format(reply_content))
            return reply

        elif context.type == ContextType.IMAGE_CREATE:
            reply = None
            return reply

    def reply_text(self, session: ChatGPTSession, api_key=None, args=None, retry_count=0) -> dict:
        """
        call openai's ChatCompletion to get the answer
        :param session: a conversation session
        :param session_id: session id
        :param retry_count: retry count
        :return: {}
        """
        try:
            # if api_key == None, the default openai.api_key will be used
            if args is None:
                args = self.args
            response = self.client.chat.completions.create( messages=session.messages, **args)
            # logger.debug("[DEEPSEEK_AI] response={}".format(response))
            # logger.info("[ChatGPT] reply={}, total_tokens={}".format(response.choices[0]['message']['content'], response["usage"]["total_tokens"]))
            return {
                "total_tokens": response.usage.total_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "content": response.choices[0].message.content,
            }
        except openai.APIError as e:
            # 处理 API 错误
            print(f"API Error: {e}")

            '''
            错误码	描述
            400 - 格式错误	原因：请求体格式错误
            解决方法：请根据错误信息提示修改请求体
            401 - 认证失败	原因：API key 错误，认证失败
            解决方法：请检查您的 API key 是否正确，如没有 API key，请先 创建 API key
            402 - 余额不足	原因：账号余额不足
            解决方法：请确认账户余额，并前往 充值 页面进行充值
            422 - 参数错误	原因：请求体参数错误
            解决方法：请根据错误信息提示修改相关参数
            429 - 请求速率达到上限	原因：请求速率（TPM 或 RPM）达到上限
            解决方法：请合理规划您的请求速率。
            500 - 服务器故障	原因：服务器内部故障
            解决方法：请等待后重试。若问题一直存在，请联系我们解决
            503 - 服务器繁忙	原因：服务器负载过高
            解决方法：请稍后重试您的请求
            '''
            #根据上面的错误码，填充下面的逻辑
            if 'status_code' in e:
                status_code = e.status_code
                if status_code == 400:
                    logger.warn("[DEEPSEEK_AI] BadRequestError: {}".format(e))
                    result["content"] = "请求体格式错误"
                    if need_retry:
                        time.sleep(20)
                elif status_code == 401:
                    logger.warn("[DEEPSEEK_AI] AuthenticationError: {}".format(e))
                    result["content"] = "API key 错误，认证失败"
                    if need_retry:
                        time.sleep(20)
                elif status_code == 402:
                    logger.warn("[DEEPSEEK_AI] BalanceError: {}".format(e))
                    result["content"] = "账号余额不足"
                    if need_retry:
                        time.sleep(20)
                elif status_code == 422:
                    logger.warn("[DEEPSEEK_AI] ParameterError: {}".format(e))
                    result["content"] = "请求体参数错误"
                    if need_retry:
                        time.sleep(20)
                elif status_code == 429:
                    logger.warn("[DEEPSEEK_AI] RateLimitError: {}".format(e))
                    result["content"] = "请求速率达到上限"
                    if need_retry:
                        time.sleep(20)
                elif status_code == 500:
                    logger.warn("[DEEPSEEK_AI] ServerError: {}".format(e))
                    result["content"] = "服务器故障"
                    if need_retry:
                        time.sleep(20)
                elif status_code == 503:
                    logger.warn("[DEEPSEEK_AI] ServerBusyError: {}".format(e))
                    result["content"] = "服务器繁忙"
                    if need_retry:
                        time.sleep(20)
            elif isinstance(e, openai.APITimeoutError):
                logger.warn("[CHATGPT] APITimeoutError: {}".format(e))
                result["content"] = "我没有收到你的消息"
                if need_retry:
                    time.sleep(20)
            elif isinstance(e, openai.APIConnectionError):
                logger.warn("[CHATGPT] APIConnectionError: {}".format(e))
                result["content"] = "我连接不到你的网络"
                if need_retry:
                    time.sleep(5)
            else:
                logger.warn("[DEEPSEEK_AI] UnknownError: {}".format(e))
                result["content"] = "未知错误: " + str(e)
                if need_retry:
                    time.sleep(20)
 
            return result
        
        except Exception as e:
            need_retry = retry_count < 2
            result = {"completion_tokens": 0, "content": "我现在有点累了，等会再来吧"}
            result["content"] = str(e)
            return result
