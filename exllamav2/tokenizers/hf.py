from typing import List, Union
from exllamav2.tokenizers.base import ExLlamaV2TokenizerBase

has_tokenizers_library = False
try:
    from tokenizers import Tokenizer
    from tokenizers import models
    has_tokenizers_library = True
except ModuleNotFoundError:
    pass

class ExLlamaV2TokenizerHF(ExLlamaV2TokenizerBase):

    space_char_: str = " "
    newline_char_: str = "\n"
    vocab = None

    def __init__(self, tokenizer_json: str) -> None:
        super().__init__()

        assert self.is_supported(), "Attempting to load HF tokenizer, but Tokenizers library is not installed"
        self.hf_tokenizer = Tokenizer.from_file(tokenizer_json)

        m = self.hf_tokenizer.model
        if isinstance(m, models.BPE):
            self.space_char_ = self.deduce_char_map(" ")  # "Ġ"
            self.newline_char_ = self.deduce_char_map("\n")  # "Ċ"


    @staticmethod
    def is_supported():
        global has_tokenizers_library
        return has_tokenizers_library

    def unk_id(self) -> int or None: return None
    def pad_id(self) -> int or None: return None
    def bos_id(self) -> int or None: return None
    def eos_id(self) -> int or None: return None
    def unk_token(self) -> str or None: return None
    def pad_token(self) -> str or None: return None
    def bos_token(self) -> str or None: return None
    def eos_token(self) -> str or None: return None

    def space_char(self): return self.space_char_
    def newline_char(self): return self.newline_char_

    def enumerate_tokens(self):
        if self.vocab is not None: return enumerate(self.vocab)
        self.vocab = []
        for i in range(self.vocab_size()):
            d = self.hf_tokenizer.decode([i])
            # p = self.hf_tokenizer.id_to_token(i)
            # if "�" in d: d = self.clean_special_chars(p)
            self.vocab.append(d)
        return enumerate(self.vocab)

    def vocab_size(self) -> int:
        return self.hf_tokenizer.get_vocab_size()

    def id_to_piece(self, idx: int) -> str:
        if idx is None: return ""
        return self.hf_tokenizer.id_to_token(idx)

    def piece_to_id(self, text: str) -> int:
        return self.hf_tokenizer.token_to_id(text)

    def decode(self, ids: List[int]) -> str:
        text = self.hf_tokenizer.decode(ids)
        return text

    def encode(self, text: list or str) -> list:
        encoding = self.hf_tokenizer.encode(text, add_special_tokens = False)
        return encoding.ids
