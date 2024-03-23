from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Tokenizer,
)

class ExLlamaV2Filter:

    # Internal state

    model: ExLlamaV2
    tokenizer: ExLlamaV2Tokenizer
    sequence_str: str


    def __init__(self,
                 model: ExLlamaV2,
                 tokenizer: ExLlamaV2Tokenizer):

        self.model = model
        self.tokenizer = tokenizer
        self.sequence_str = ""


    def clone(self, c = None):
        if c is None:
            c = ExLlamaV2Filter.__new__(ExLlamaV2Filter)
        c.model = self.model
        c.tokenizer = self.tokenizer
        c.sequence_str = self.sequence_str
        return c


    def begin(self, prefix_str):
        pass


    def feed(self, token):
        pass


    def next(self):
        pass

