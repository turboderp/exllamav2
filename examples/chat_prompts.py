
class PromptFormat:

    botname = "Chatbort"
    username = "User"

    def __init__(self):
        pass

    #

    def default_system_prompt(self):
        raise NotImplementedError

    def first_prompt(self):
        raise NotImplementedError

    def subs_prompt(self):
        raise NotImplementedError

    def stop_conditions(self, tokenizer):
        raise NotImplementedError

    def encoding_options(self):  # (add_bos, add_eos, encode_special_tokens)
        raise NotImplementedError

    def print_bot_name(self):
        return False

    def print_extra_newline(self):
        return False


class PromptFormat_raw(PromptFormat):

    description = "Model-agnostic mode simulating a raw chatlog"

    def __init__(self):
        super().__init__()
        pass

    def default_system_prompt(self):
        return \
            f"""This is a conversation between a helpful AI assistant named {self.botname} and a """ + \
            (f"""user named {self.username}.""" if self.username != "User" else """user.""")

    def first_prompt(self):
        return \
            f"""<|system_prompt|>\n{self.username}: <|user_prompt|>\n{self.botname}:"""

    def subs_prompt(self):
        return \
            f"""{self.username}: <|user_prompt|>\n{self.botname}:"""

    def stop_conditions(self, tokenizer):
        return \
            [self.username + ":",
             self.username[0:1] + ":",
             self.username.upper() + ":",
             self.username.lower() + ":",
             tokenizer.eos_token_id]

    def encoding_options(self):
        return False, False, False

    def print_bot_name(self):
        return True


class PromptFormat_llama(PromptFormat):

    description = "Llama-chat, Llama2-chat and Mistral-instruct models"

    def __init__(self):
        super().__init__()
        pass

    def default_system_prompt(self):
        return \
            """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  """ + \
            """Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. """ + \
            """Please ensure that your responses are socially unbiased and positive in nature."""

    def first_prompt(self):
        return \
            """[INST] <<SYS>>\n<|system_prompt|>\n<</SYS>>\n\n<|user_prompt|> [/INST]"""

    def subs_prompt(self):
        return \
            """[INST] <|user_prompt|> [/INST]"""

    def stop_conditions(self, tokenizer):
        return \
            [tokenizer.eos_token_id]

    def encoding_options(self):
        return True, False, False

    def print_extra_newline(self):
        return True


class PromptFormat_codellama(PromptFormat_llama):

    description = "CodeLlama-instruct"

    def __init__(self):
        super().__init__()
        pass

    def default_system_prompt(self):
        return \
            """You are a helpful coding assistant. Always answer as helpfully as possible."""


class PromptFormat_chatml(PromptFormat):

    description = "ChatML format, as used by e.g. (Mistral)Orca"

    def __init__(self):
        super().__init__()
        pass

    def default_system_prompt(self):
        return \
            f"""You are {self.botname}, a large language model. Answer as concisely as possible."""

    def first_prompt(self):
        return \
            """<|im_start|>system\n""" + \
            """<|system_prompt|>\n""" + \
            """<|im_end|>\n""" + \
            """<|im_start|>user\n""" + \
            """<|user_prompt|><|im_end|>\n""" + \
            """<|im_start|>assistant\n"""

    def subs_prompt(self):
        return \
            """<|im_end|>\n""" + \
            """<|im_start|>user\n""" + \
            """<|user_prompt|><|im_end|>\n""" + \
            """<|im_start|>assistant\n"""

    def stop_conditions(self, tokenizer):
        return \
            [tokenizer.eos_token_id,
             """<|im_end|>"""]

    def encoding_options(self):
        return False, False, True

    def print_extra_newline(self):
        return True


class PromptFormat_tinyllama(PromptFormat_chatml):

    description = "ChatML format, but ignoring special/added tokens. Use for TinyLlama-chat v0.3"

    def encoding_options(self):
        return False, False, False


class PromptFormat_zephyr(PromptFormat):

    description = "Zephyr 7b alpha prompt format."

    def __init__(self):
        super().__init__()
        pass

    def default_system_prompt(self):
        return \
            f"""You are {self.botname}, a large language model. Answer as concisely as possible."""

    def first_prompt(self):
        return \
            """<|system|>\n""" + \
            """<|system_prompt|>\n""" + \
            """</s>\n""" + \
            """<|user|>\n""" + \
            """<|user_prompt|></s>\n""" + \
            """<|assistant|>\n"""

    def subs_prompt(self):
        return \
            """<|user|>\n""" + \
            """<|user_prompt|></s>\n""" + \
            """<|assistant|>\n"""

    def stop_conditions(self, tokenizer):
        return \
            [tokenizer.eos_token_id,
             """<|user|>""",
             """</s>"""]

    def encoding_options(self):
        return False, False, True

    def print_extra_newline(self):
        return True

prompt_formats = \
{
    "raw": PromptFormat_raw,
    "llama": PromptFormat_llama,
    "codellama": PromptFormat_codellama,
    "chatml": PromptFormat_chatml,
    "tinyllama": PromptFormat_tinyllama,
    "zephyr": PromptFormat_zephyr
}






