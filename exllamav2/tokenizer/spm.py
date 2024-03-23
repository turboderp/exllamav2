from __future__ import annotations
from typing import List, Union
from sentencepiece import SentencePieceProcessor
from exllamav2.tokenizer.base import ExLlamaV2TokenizerBase

# Wrapper for SentencePiece

class ExLlamaV2TokenizerSPM(ExLlamaV2TokenizerBase):

    vocab: list[str] | None

    def __init__(self, tokenizer_model: str):
        super().__init__()
        self.vocab = None
        self.spm = SentencePieceProcessor(model_file = tokenizer_model)

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
        if self.vocab is not None: return enumerate(self.vocab)
        self.vocab = []
        for i in range(self.vocab_size()):
            p = self.spm.id_to_piece(i)
            if all(c == self.space_char() for c in p):
                d = " " * len(p)
            else:
                d = self.spm.decode(i)
                if p.startswith(self.space_char()) and not d.startswith(" "): d = " " + d
            self.vocab.append(d)
        return enumerate(self.vocab)

    def id_to_piece(self, idx: int) -> str:
        return self.spm.id_to_piece(idx)

    def piece_to_id(self, text: str) -> int:
        return self.spm.piece_to_id(text)

    def vocab_size(self) -> int:
        return self.spm.vocab_size()

    def decode(self, ids: List[int]) -> str:
        text = self.spm.decode(ids)
        return text

    def encode(self, text: list or str) -> list:
        encoding = self.spm.EncodeAsIds(text)
        return encoding
