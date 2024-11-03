from bot.session_manager import Session
from common.log import logger
from common import const

"""
    e.g.  [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]
"""


class DeepSeekSession(Session):
    def __init__(self, session_id, system_prompt=None, model="gpt-3.5-turbo"):
        super().__init__(session_id, system_prompt)
        self.model = model
        self.reset()

    def __str__(self):
        # 构造对话模型的输入
        """
        e.g.  Q: xxx
              A: xxx
              Q: xxx
        """
        prompt = ""
        for item in self.messages:
            if item["role"] == "system":
                prompt += item["content"] + "<|endoftext|>\n\n\n"
            elif item["role"] == "user":
                prompt += "Q: " + item["content"] + "\n"
            elif item["role"] == "assistant":
                prompt += "\n\nA: " + item["content"] + "<|endoftext|>\n"

        if len(self.messages) > 0 and self.messages[-1]["role"] == "user":
            prompt += "A: "
        return prompt

    def discard_exceeding(self, max_tokens, cur_tokens=None):
        precise = True
        try:
            cur_tokens = self.calc_tokens()
        except Exception as e:
            precise = False
            if cur_tokens is None:
                raise e
            logger.debug("Exception when counting tokens precisely for query: {}".format(e))
        while cur_tokens > max_tokens:
            if len(self.messages) > 2:
                self.messages.pop(1)
            elif len(self.messages) == 2 and self.messages[1]["role"] == "assistant":
                self.messages.pop(1)
                if precise:
                    cur_tokens = self.calc_tokens()
                else:
                    cur_tokens = cur_tokens - max_tokens
                break
            elif len(self.messages) == 2 and self.messages[1]["role"] == "user":
                logger.warn("user message exceed max_tokens. total_tokens={}".format(cur_tokens))
                break
            else:
                logger.debug("max_tokens={}, total_tokens={}, len(messages)={}".format(max_tokens, cur_tokens, len(self.messages)))
                break
            if precise:
                cur_tokens = self.calc_tokens()
            else:
                cur_tokens = cur_tokens - max_tokens
        return cur_tokens

    def calc_tokens(self):
        return num_tokens_from_string(str(self), self.model)


# refer to https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_string(string: str, model: str) -> int:
    """Returns the number of tokens in a text string."""
    
    import transformers
    chat_tokenizer_dir = "./"

    tokenizer = transformers.AutoTokenizer.from_pretrained( 
            chat_tokenizer_dir, trust_remote_code=True
            )

    result = tokenizer.encode("Hello!")
    print(result)
    return len(result)

