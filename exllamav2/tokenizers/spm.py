from typing import List, Union
from sentencepiece import SentencePieceProcessor
from exllamav2.tokenizers.base import ExLlamaV2TokenizerBase

class ExLlamaV2TokenizerSPM(ExLlamaV2TokenizerBase):

    def __init__(self, tokenizer_model: str):
        super().__init__()
        self.spm = SentencePieceProcessor(model_file = model_file)

    def unk_id(self) -> int or None: return self.spm.unk_id()
    def pad_id(self) -> int or None: return self.spm.pad_id()
    def bos_id(self) -> int or None: return self.spm.bos_id()
    def eos_id(self) -> int or None: return self.spm.eos_id()
    def unk_token(self) -> str or None: return None
    def pad_token(self) -> str or None: return None
    def bos_token(self) -> str or None: return None
    def eos_token(self) -> str or None: return None

    def space_char(self): return "▁"
    def newline_char(self): return "\n"

    def enumerate_tokens(self):
        all_tokens = list(range(self.vocab_size()))
        return enumerate(self.spm.id_to_piece(all_tokens))

    def id_to_piece(self, idx: int) -> str: raise NotImplementedError()

    def piece_to_id(self, text: str) -> int: raise NotImplementedError()

    def vocab_size(self) -> int:
        return self.spm.vocab_size()

    def decode(self, idx: int) -> str:
        return self.spm.decode(idx)

    def Decode(self, ids: List[int]) -> str:
        return self.spm.Decode(ids)

    def Encode(self, text: str) -> list:
        return self.spm.Encode(text)

    def EncodeAsIds(self, text: str) -> list:
        return self.spm.EncodeAsIds(text)
