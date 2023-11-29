from typing import List, Union
import re

class ExLlamaV2TokenizerBase:

    ord_exp = re.compile(r"^<0x([0-9A-Fa-f]+)>$")

    def __init__(self):
        pass

    def unk_id(self) -> int or None: raise NotImplementedError()
    def pad_id(self) -> int or None: raise NotImplementedError()
    def bos_id(self) -> int or None: raise NotImplementedError()
    def eos_id(self) -> int or None: raise NotImplementedError()
    def unk_token(self) -> str or None: raise NotImplementedError()
    def pad_token(self) -> str or None: raise NotImplementedError()
    def bos_token(self) -> str or None: raise NotImplementedError()
    def eos_token(self) -> str or None: raise NotImplementedError()

    def space_char(self) -> str: raise NotImplementedError()
    def newline_char(self) -> str: raise NotImplementedError()

    def enumerate_tokens(self): raise NotImplementedError()
    def vocab_size(self) -> int: raise NotImplementedError()

    def id_to_piece(self, idx: int) -> str: raise NotImplementedError()
    def piece_to_id(self, text: str) -> int: raise NotImplementedError()
    def decode(self, ids: list) -> str: raise NotImplementedError()
    def encode(self, text: list or str) -> list: raise NotImplementedError()

    def clean_special_chars(self, p):
        p = p.replace(self.space_char(), " ")
        p = p.replace(self.newline_char(), "\n")
        return p

    def piece_to_ord(self, p):
        match = self.ord_exp.match(p)
        if match:
            h = match.group(1)
            return int(h, 16)
        if len(p) == 1:
            p = self.clean_special_chars(p)
            o = ord(p)
            if o <= 255: return o
        return -1

    def id_to_ord(self, idx: int) -> int:
        piece = self.id_to_piece(idx)
        return self.piece_to_ord(piece)

    def deduce_char_map(self, input_char):
        char_id = self.encode(input_char)[-1]
        char_str = self.id_to_piece(char_id)
        match = self.ord_exp.match(char_str)
        if match:
            h = match.group(1)
            o = int(h, 16)
            char_str = chr(o)
        else:
            char_str = char_str[-1]
        return char_str

