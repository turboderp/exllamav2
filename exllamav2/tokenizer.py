from exllamav2.config import ExLlamaV2Config
from sentencepiece import SentencePieceProcessor
from transformers import AutoTokenizer
from typing import List, Union
import torch
import os, json, re
import warnings

class BaseTokenizer:
    def __init__(self, model) -> None:
        self.tokenizer_model = model

    def unk_id(self) -> int:
        return self.tokenizer_model.unk_id()
    
    def pad_id(self) -> int:
        return self.tokenizer_model.pad_id()
    
    def pad_token(self) -> str:
        return None
    
    def bos_id(self) -> int:
        return self.tokenizer_model.bos_id()
    
    def bos_token(self) -> str:
        return None

    def eos_id(self) -> int:
        return self.tokenizer_model.eos_id()
    
    def eos_token(self) -> str:
        return None
    
    def vocab_size(self) -> int:
        return self.tokenizer_model.vocab_size()
    
    def id_to_piece(self, idx: Union[int, List[int]]) -> str:
        return self.tokenizer_model.id_to_piece(idx)
    
    def decode(self, idx: int) -> str:
        return self.tokenizer_model.decode(idx)
    
    def Decode(self, ids: List[int]) -> str:
        return self.tokenizer_model.Decode(ids)
    
    def Encode(self, text: str) -> list:
        return self.tokenizer_model.Encode(text)
    
    def EncodeAsIds(self, text: str) -> list:
        return self.tokenizer_model.EncodeAsIds(text)


class SentencePieceTokenizer(BaseTokenizer):
    def __init__(self, model_file:str) -> None:
        super().__init__(SentencePieceProcessor(model_file=model_file))


class HuggingFaceAutoTokenizer(BaseTokenizer):
    def __init__(self, model_name:str) -> None:
        super().__init__(AutoTokenizer.from_pretrained(model_name, trust_remote_code=True))
    
    def unk_id(self) -> int:
        return self.tokenizer_model.unk_token_id
    
    def eos_id(self) -> int:
        return self.tokenizer_model.eos_token_id
    
    def eos_token(self) -> str:
        return self.tokenizer_model.eos_token
    
    def bos_id(self) -> int:
        return self.tokenizer_model.bos_token_id
    
    def bos_token(self) -> str:
        return self.tokenizer_model.bos_token
    
    def pad_id(self) -> int:
        return self.tokenizer_model.pad_token_id
    
    def pad_token(self) -> str:
        return self.tokenizer_model.pad_token
    
    def vocab_size(self) -> int:
        return self.tokenizer_model.vocab_size
    
    def id_to_piece(self, idx: Union[int, List[int]]) -> str:
        if isinstance(idx, int):
            return self.tokenizer_model.decode(idx)
        elif isinstance(idx, list):
            return [self.tokenizer_model.decode(i) for i in idx]
    
    def decode(self, id: int) -> str:
        return self.tokenizer_model.decode(id)
    
    def Decode(self, ids: List[int]) -> str:
        return self.tokenizer_model.decode(ids)
    
    def Encode(self, text: str) -> list:
        return self.tokenizer_model.encode(text)
    
    def EncodeAsIds(self, text: str) -> list:
        if self.tokenizer_model.is_fast or not self.tokenizer_model.split_special_tokens:
            warnings.warn("Current HuggingFace tokenizer doesn't support encode ignoring special token")
        return self.tokenizer_model.encode(text)


class ExLlamaV2Tokenizer:

    class Trie:

        children: dict = {}
        leaf: list = []

        def __init__(self, children = None, leaf = None):
            self.children = children if children is not None else {}
            self.leaf = leaf if leaf is not None else []


    config: ExLlamaV2Config
    tokenizer: BaseTokenizer

    unk_token: str = "<unk>"
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    pad_token: str = ""
    newline_token: str = "\n"
    unk_token_id: int
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    newline_token_id: int = 13

    ord_exp = re.compile(r"^<0x([0-9A-Fa-f]+)>$")
    id_to_ord: list = None
    id_to_piece: list = None
    piece_to_id: dict = None
    prefix_to_ids: dict = None
    prefix_id_to_ids: dict = None
    char_trie: Trie = None
    char_trie_ci: Trie = None

    extended_id_to_piece = {}
    extended_piece_to_id = {}
    special_delimiters = None

    tokenized_str_cache = {}
    max_cached_strings = 100

    def __init__(self, config, lazy_init = False):

        self.config = config

        self.tokenizer = SentencePieceTokenizer(model_file = self.config.tokenizer_path) \
            if os.path.exists(self.config.tokenizer_path) \
            else HuggingFaceAutoTokenizer(model_name = os.path.dirname(self.config.tokenizer_path))

        # Load added_tokens.json if present

        added_tokens_path = os.path.join(self.config.model_dir, "added_tokens.json")
        if os.path.exists(added_tokens_path):
            with open(added_tokens_path) as f:
                self.extended_piece_to_id = json.load(f)

        self.extended_id_to_piece = { v: k for k, v in self.extended_piece_to_id.items() }

        # Get control token IDs

        # self.eos_token_id = self.tokenizer.eos_id()
        # self.bos_token_id = self.tokenizer.bos_id()
        # self.unk_token_id = config.unk_token_id

        self.unk_token_id = self.tokenizer.unk_id()
        self.eos_token_id = config.eos_token_id
        self.bos_token_id = config.bos_token_id

        # Get control token strings

        try: self.unk_token = (self.tokenizer.unk_token() or self.extended_id_to_piece.get(self.unk_token_id, None)) or self.tokenizer.id_to_piece(self.unk_token_id)
        except: pass
        try: self.bos_token = (self.tokenizer.bos_token() or self.extended_id_to_piece.get(self.bos_token_id, None)) or self.tokenizer.id_to_piece(self.bos_token_id)
        except: pass
        try: self.eos_token = (self.tokenizer.eos_token() or self.extended_id_to_piece.get(self.eos_token_id, None)) or self.tokenizer.id_to_piece(self.eos_token_id)
        except: pass

        self.pad_token_id = 0

        # Special case if <unk> and <pad> have the same ID

        if self.unk_token_id == self.pad_token_id:
            self.unk_token = self.pad_token

        # Make sure extended vocab contains control tokens, but avoid empty pieces

        if isinstance(self.tokenizer, SentencePieceTokenizer):

            if self.unk_token != "":
                self.extended_piece_to_id[self.unk_token] = self.unk_token_id
                self.extended_id_to_piece[self.unk_token_id] = self.unk_token
            if self.bos_token != "":
                self.extended_piece_to_id[self.bos_token] = self.bos_token_id
                self.extended_id_to_piece[self.bos_token_id] = self.bos_token
            if self.eos_token != "":
                self.extended_piece_to_id[self.eos_token] = self.eos_token_id
                self.extended_id_to_piece[self.eos_token_id] = self.eos_token

        # Create dictionaries on init

        if not lazy_init:

            self.get_id_to_ord_list()
            self.get_id_to_piece_list()
            self.get_piece_to_id_dict()
            self.get_prefix_to_ids_dict()
            self.get_prefix_id_to_ids_dict()
            self.get_char_trie()
            self.get_char_trie_ci()


    # Get single token

    def single_token(self, token_id: int):

        return torch.tensor([[token_id]], dtype = torch.long)


    # Encode string with special tokens

    def encode_special(self, text: str):

        if isinstance(self.tokenizer, SentencePieceTokenizer):
            if self.special_delimiters is None:
                self.special_delimiters = re.compile("(" + "|".join(map(re.escape, self.extended_piece_to_id.keys())) + ")")

            split = self.special_delimiters.split(text)
            encoded = []

            i = 0
            while i < len(split):
                if split[i] != "": encoded += self.tokenizer.EncodeAsIds(split[i])
                if i + 1 < len(split): encoded += [self.extended_piece_to_id[split[i + 1]]]
                i += 2
        else:
            encoded = self.tokenizer.Encode(text)
        return encoded


    # Encode string

    # TODO: Deal with rstrip and lstrip for added tokens

    def encode(self, text, add_bos = False, add_eos = False, encode_special_tokens = False):

        if isinstance(text, list):

            # text is a list of strings

            if encode_special_tokens:
                list_ids = [self.encode_special(t) for t in text]
            else:
                list_ids = self.tokenizer.EncodeAsIds(text)

            if add_bos:
                for ids in list_ids: ids.insert(0, self.bos_token_id)
            if add_eos:
                for ids in list_ids: ids.append(self.eos_token_id)

            max_length = max([len(ids) for ids in list_ids])

            padded_ids = []
            for ids in list_ids:
                padding = torch.full((max_length - len(ids),), self.pad_token_id)
                sequence = torch.tensor(ids)
                padded_ids.append(torch.cat((padding, sequence), dim = 0))

            return torch.stack(padded_ids, dim = 0)

        else:

            # text is a single string

            if encode_special_tokens:
                ids = self.encode_special(text)
            else:
                ids = self.tokenizer.EncodeAsIds(text)

            if add_bos:
                ids.insert(0, self.bos_token_id)
            if add_eos:
                ids.append(self.eos_token_id)

            return torch.tensor(ids).to(torch.long).unsqueeze(0)


    # Decode sequence with or without special tokens

    def decode_(self, seq, decode_special_tokens):

        if not decode_special_tokens:

            max_token = self.tokenizer.vocab_size()
            seq = [t for t in seq if (t != self.pad_token_id and t < max_token and t!= self.eos_token_id)]
            if self.eos_token_id in seq: seq = seq[:seq.index(self.eos_token_id)]
            return self.tokenizer.Decode(seq)

        else:

            text = ""
            start = 0
            end = 0
            while end < len(seq):
                if seq[end] in self.extended_id_to_piece:
                    if end > start: text += self.tokenizer.Decode(seq[start : end])
                    text += self.extended_id_to_piece[seq[end]]
                    end += 1
                    start = end
                else:
                    end += 1
            if end > start: text += self.tokenizer.Decode(seq[start : end])

        return text


    # Decode IDs

    def decode(self, ids, decode_special_tokens = False):

        if ids.dim() > 1:

            texts = []
            for i in range(ids.shape[0]):
                seq = ids[i].tolist()
                texts.append(self.decode_(seq, decode_special_tokens))
            return texts

        else:

            ids = ids.tolist()
            text = self.decode_(ids, decode_special_tokens)
            return text


    # Create padding mask

    def padding_mask(self, ids):

        mask = (ids == self.pad_token_id)
        mask = mask.int()
        mask *= -65504
        mask = mask.half()
        return mask


    def num_tokens(self, text):

        ids = self.tokenizer.Encode(text)
        return len(ids)


    # Get ordinals of single-strings tokens

    def get_id_to_ord_list(self):

        if self.id_to_ord is not None: return self.id_to_ord

        all_tokens = list(range(self.tokenizer.vocab_size()))
        self.id_to_ord = []
        for idx, p in enumerate(self.tokenizer.id_to_piece(all_tokens)):
            match = self.ord_exp.match(p)
            if match:
                h = match.group(1)
                o = int(h, 16)
            elif len(p) == 1:
                o = ord(p)
                if o > 255: o = -1
            else:
                o = -1
            self.id_to_ord.append(o)

        i = self.tokenizer.vocab_size()
        while i in self.extended_id_to_piece:
            self.id_to_ord.append(-1)
            i += 1


    # Copy vocabulary from SP model

    def get_id_to_piece_list(self):

        if self.id_to_piece is not None: return self.id_to_piece

        all_tokens = list(range(self.tokenizer.vocab_size()))
        self.id_to_piece = \
        [
            (p.replace("▁", " ") if not p.startswith("<") else self.tokenizer.decode(idx))
            for idx, p in enumerate(self.tokenizer.id_to_piece(all_tokens))
        ]

        i = self.tokenizer.vocab_size()
        while i in self.extended_id_to_piece:
            self.id_to_piece.append(self.extended_id_to_piece[i])
            i += 1

        return self.id_to_piece


    def get_piece_to_id_dict(self):

        if self.piece_to_id is not None: return self.piece_to_id

        all_pieces = self.get_id_to_piece_list()
        self.piece_to_id = { piece: idx for idx, piece in enumerate(all_pieces) }
        return self.piece_to_id


    # Create dictionary mapping prefixes to token IDs

    def get_prefix_to_ids_dict(self):

        if self.prefix_to_ids is not None: return self.prefix_to_ids

        piece_to_id = self.get_piece_to_id_dict()
        pieces = sorted(list(piece_to_id.keys()))
        pieces = [p for p in pieces if len(p) > 0]
        self.prefix_to_ids = {}

        for i in range(len(pieces)):
            piece = pieces[i]
            if len(piece) == 0: continue
            piece_id = piece_to_id[pieces[i]]
            self.prefix_to_ids[piece] = [piece_id]

            for j in range(1, len(piece)):
                fpiece = piece[:-j]
                if fpiece in self.prefix_to_ids:
                    self.prefix_to_ids[fpiece].append(piece_id)

        self.prefix_to_ids = { prefix: sorted(ids) for prefix, ids in self.prefix_to_ids.items() }

        return self.prefix_to_ids


    # Create dictionary mapping each ID to any IDs that it prefixes

    def get_prefix_id_to_ids_dict(self):

        if self.prefix_id_to_ids is not None: return self.prefix_id_to_ids

        piece_to_id = self.get_piece_to_id_dict()
        prefix_to_ids = self.get_prefix_to_ids_dict()

        self.prefix_id_to_ids = { piece_to_id[piece]: ids for piece, ids in prefix_to_ids.items() }

        for i in range(self.config.vocab_size):
            if i not in self.prefix_id_to_ids:
                self.prefix_id_to_ids[i] = [i]

        return self.prefix_id_to_ids


    # Create trie mapping chars to token IDs

    def _make_trie(self, ci):

        id_to_piece = self.get_id_to_piece_list()
        trie = ExLlamaV2Tokenizer.Trie()

        for idx, piece in enumerate(id_to_piece):

            if ci: piece = piece.lower()

            w = trie
            while piece != "":

                p = piece[0]
                piece = piece[1:]

                if p not in w.children: w.children[p] = ExLlamaV2Tokenizer.Trie()
                w = w.children[p]

                if piece == "": w.leaf.append(idx)

        return trie


    def get_char_trie(self):

        if self.char_trie is not None: return self.char_trie

        self.char_trie = self._make_trie(False)
        return self.char_trie


    def get_char_trie_ci(self):

        if self.char_trie_ci is not None: return self.char_trie_ci

        self.char_trie_ci = self._make_trie(True)
        return self.char_trie_ci


    # Cached tokenization

    def cached_encode_str(self, text: str):

        if text in self.tokenized_str_cache:
            return self.tokenized_str_cache[text]

        while len(self.tokenized_str_cache) >= self.max_cached_strings:
            del self.tokenized_str_cache[next(iter(self.tokenized_str_cache))]  # Always removes oldest entry as of Python 3.7

        new_enc = self.encode(text)
        self.tokenized_str_cache[text] = new_enc
        return new_enc
