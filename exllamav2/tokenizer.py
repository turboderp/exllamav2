from exllamav2.config import ExLlamaV2Config
import torch
import os, json, re
from exllamav2.tokenizers import (
    ExLlamaV2TokenizerBase,
    ExLlamaV2TokenizerSPM,
    ExLlamaV2TokenizerHF
)

class ExLlamaV2Tokenizer:

    class Trie:

        children: dict = {}
        leaf: list = []

        def __init__(self, children = None, leaf = None):
            self.children = children if children is not None else {}
            self.leaf = leaf if leaf is not None else []


    config: ExLlamaV2Config
    tokenizer: ExLlamaV2TokenizerBase

    unk_token: str = "<unk>"
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    pad_token: str = ""
    newline_token: str = "\n"
    space_token: str = " "
    unk_token_id: int
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    newline_token_id: int or None
    space_token_id: int or None

    id_to_ord: list = None
    id_to_piece: list = None
    piece_to_id: dict = None
    prefix_to_ids: dict = None
    prefix_id_to_ids: dict = None
    char_trie: Trie = None
    char_trie_ci: Trie = None

    unspecial_piece_to_id = {}
    unspecial_id_to_piece = {}
    extended_id_to_piece = {}
    extended_piece_to_id = {}
    special_delimiters = None
    unspecial_delimiters = None

    tokenized_str_cache = {}
    max_cached_strings = 100

    def __init__(self, config, lazy_init = False, force_json = False):

        self.config = config

        # Detect tokenizer type and initialize

        path_spm = os.path.join(self.config.model_dir, "tokenizer.model")
        path_hf = os.path.join(self.config.model_dir, "tokenizer.json")

        if os.path.exists(path_spm) and not force_json: self.tokenizer = ExLlamaV2TokenizerSPM(path_spm)
        elif os.path.exists(path_hf): self.tokenizer = ExLlamaV2TokenizerHF(path_hf)
        else: raise FileNotFoundError("No supported tokenizer found.")

        # Attempt to load added tokens from tokenizer.json
        # TODO: Deal with rstrip and lstrip for added, non-control tokens

        self. unspecial_piece_to_id = {}

        tokenizer_json_path = os.path.join(self.config.model_dir, "tokenizer.json")
        if os.path.exists(tokenizer_json_path):
            with open(tokenizer_json_path, encoding = "utf8") as f:
                tokenizer_json = json.load(f)
                if "added_tokens" in tokenizer_json:
                    for v in tokenizer_json["added_tokens"]:
                        if v["special"]:
                            self.extended_piece_to_id[v["content"]] = v["id"]
                        else:
                            self.unspecial_piece_to_id[v["content"]] = v["id"]

        # Add tokens from added_tokens.json if present, assume they're all special

        added_tokens_path = os.path.join(self.config.model_dir, "added_tokens.json")
        if os.path.exists(added_tokens_path):
            with open(added_tokens_path, encoding = "utf8") as f:
                self.extended_piece_to_id = json.load(f)

        # Remove unspecial added tokens that exist in the base tokenizer already, but only if they decode correctly
        # see https://github.com/huggingface/tokenizers/issues/1392

        ok_tokens = []
        for p, i in self.unspecial_piece_to_id.items():
            try:
                itp = self.tokenizer.decode([i])
                if itp == p: ok_tokens.append(p)
            except IndexError:
                pass
        for t in ok_tokens: del self.unspecial_piece_to_id[t]

        # Invert extended dictionaries

        self.extended_id_to_piece = { v: k for k, v in self.extended_piece_to_id.items() }
        self.unspecial_id_to_piece = { v: k for k, v in self.unspecial_piece_to_id.items() }

        # Get control token IDs

        self.unk_token_id = self.tokenizer.unk_id()
        self.eos_token_id = config.eos_token_id
        self.bos_token_id = config.bos_token_id

        # Get control token strings

        self.unk_token = (self.tokenizer.unk_token() or self.extended_id_to_piece.get(self.unk_token_id, None)) or self.tokenizer.id_to_piece(self.unk_token_id)
        self.bos_token = (self.tokenizer.bos_token() or self.extended_id_to_piece.get(self.bos_token_id, None)) or self.tokenizer.id_to_piece(self.bos_token_id)
        self.eos_token = (self.tokenizer.eos_token() or self.extended_id_to_piece.get(self.eos_token_id, None)) or self.tokenizer.id_to_piece(self.eos_token_id)

        # Some tokenizers use token ID zero for text but don't explicitly define a padding token but provide one anyway

        pad_test = self.tokenizer.piece_to_id("<pad>")
        self.pad_token_id = pad_test or self.eos_token_id

        # Special case if <unk> and <pad> have the same ID

        if self.unk_token_id == self.pad_token_id:
            self.unk_token = self.pad_token

        # Make sure extended vocab contains control tokens, but avoid empty pieces

        if self.unk_token:
            self.extended_piece_to_id[self.unk_token] = self.unk_token_id
            self.extended_id_to_piece[self.unk_token_id] = self.unk_token
        if self.bos_token:
            self.extended_piece_to_id[self.bos_token] = self.bos_token_id
            self.extended_id_to_piece[self.bos_token_id] = self.bos_token
        if self.eos_token:
            self.extended_piece_to_id[self.eos_token] = self.eos_token_id
            self.extended_id_to_piece[self.eos_token_id] = self.eos_token

        # Useful token IDs

        try: self.newline_token_id = self.tokenizer.encode(self.newline_token)[-1]
        except: self.newline_token_id = None
        try: self.space_token_id = self.tokenizer.encode(self.space_token)[-1]
        except: self.space_token_id = None

        # Create dictionaries on init

        if not lazy_init:

            self.get_id_to_ord_list()
            self.get_id_to_piece_list()
            self.get_piece_to_id_dict()
            self.get_prefix_to_ids_dict()
            self.get_prefix_id_to_ids_dict()
            self.get_char_trie()
            self.get_char_trie_ci()


    # Return size of valid vocabulary

    def get_vocab_size(self):

        id_to_piece = self.get_id_to_piece_list()
        return len(id_to_piece)


    # Get single token

    def single_token(self, token_id: int):

        return torch.tensor([[token_id]], dtype = torch.long)


    # Encode string with added, unspecial tokens

    def encode_unspecial(self, text: str):

        if not self.unspecial_piece_to_id:
            return self.tokenizer.encode(text)

        if self.unspecial_delimiters is None:
            self.unspecial_delimiters = re.compile("(" + "|".join(map(re.escape, self.unspecial_piece_to_id.keys())) + ")")

        split = self.unspecial_delimiters.split(text)
        encoded = []

        i = 0
        while i < len(split):
            if split[i] != "": encoded += self.tokenizer.encode(split[i])
            if i + 1 < len(split): encoded += [self.unspecial_piece_to_id[split[i + 1]]]
            i += 2

        return encoded


    # Encode string with special tokens

    def encode_special(self, text: str):

        if self.special_delimiters is None:
            self.special_delimiters = re.compile("(" + "|".join(map(re.escape, self.extended_piece_to_id.keys())) + ")")

        split = self.special_delimiters.split(text)
        encoded = []

        i = 0
        while i < len(split):
            if split[i] != "": encoded += self.tokenizer.encode(split[i])
            if i + 1 < len(split): encoded += [self.extended_piece_to_id[split[i + 1]]]
            i += 2

        return encoded


    # Encode string
    # TODO: Deal with rstrip and lstrip for control tokens

    def encode(self, text, add_bos = False, add_eos = False, encode_special_tokens = False, return_offsets = False):

        if isinstance(text, list):

            # text is a list of strings

            list_ids = [self.encode_special(t) for t in text] if encode_special_tokens else [self.encode_unspecial(t) for t in text]

            if add_bos:
                for ids in list_ids: ids.insert(0, self.bos_token_id)
            if add_eos:
                for ids in list_ids: ids.append(self.eos_token_id)

            max_length = max([len(ids) for ids in list_ids])

            padded_ids = []
            offsets = []
            for ids in list_ids:
                padding_length = max_length - len(ids)
                padding = torch.full((padding_length,), self.pad_token_id)
                padded_ids.append(torch.cat((padding, torch.tensor(ids)), dim = 0))
                offsets.append(-padding_length)

            stacked_ids = torch.stack(padded_ids, dim=0)

            if return_offsets:
                return stacked_ids, torch.tensor(offsets, dtype = torch.int)
            else:
                return stacked_ids

        else:

            # text is a single string

            ids = self.encode_special(text) if encode_special_tokens else self.encode_unspecial(text)
            if add_bos: ids.insert(0, self.bos_token_id)
            if add_eos: ids.append(self.eos_token_id)

            ids = torch.tensor(ids).to(torch.long).unsqueeze(0)
            if return_offsets:
                return ids, torch.tensor([0], dtype = torch.int)
            else:
                return ids


    # Decode sequence with added, unspecial tokens

    def decode_unspecial(self, seq):

        if not self.unspecial_id_to_piece:
            return self.tokenizer.decode(seq)

        text = ""
        start = 0
        end = 0
        while end < len(seq):
            if seq[end] in self.unspecial_id_to_piece:
                if end > start: text += self.tokenizer.decode(seq[start: end])
                text += self.unspecial_id_to_piece[seq[end]]
                end += 1
                start = end
            else:
                end += 1
        if end > start: text += self.tokenizer.decode(seq[start: end])
        return text



    # Decode sequence with or without special tokens

    def decode_(self, seq, decode_special_tokens):

        if not decode_special_tokens:

            max_token = self.tokenizer.vocab_size()
            seq = [t for t in seq if (t != self.pad_token_id and t < max_token and t != self.eos_token_id)]
            if self.eos_token_id in seq: seq = seq[:seq.index(self.eos_token_id)]
            return self.decode_unspecial(seq)

        else:

            text = ""
            start = 0
            end = 0
            while end < len(seq):
                if seq[end] in self.extended_id_to_piece:
                    if end > start: text += self.tokenizer.decode(seq[start : end])
                    text += self.extended_id_to_piece[seq[end]]
                    end += 1
                    start = end
                else:
                    end += 1
            if end > start: text += self.decode_unspecial(seq[start : end])

        return text


    # Decode IDs, or a list of IDs

    def decode(self, ids, decode_special_tokens = False):

        if isinstance(ids, list):

            texts = []
            for i in ids:
                texts.append(self.decode(i, decode_special_tokens))
            return texts

        assert isinstance(ids, torch.Tensor), "ids must be Tensor"

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

        mask_bool = (ids == self.pad_token_id)
        mask = mask_bool.int()
        mask *= -65505 * 2
        mask = mask.half()
        return mask


    def num_tokens(self, text):

        ids = self.tokenizer.encode(text)
        return len(ids)


    # Get ordinals of single-byte tokens

    def get_id_to_ord_list(self):

        if self.id_to_ord is not None: return self.id_to_ord

        self.id_to_ord = []
        for idx in range(self.tokenizer.vocab_size()):
            p = self.tokenizer.id_to_piece(idx)
            self.id_to_ord.append(self.tokenizer.piece_to_ord(p))

        i = self.tokenizer.vocab_size()
        while True:
            if i in self.extended_id_to_piece:
                self.id_to_ord.append(self.tokenizer.piece_to_ord(self.extended_id_to_piece[i]))
            elif i in self.unspecial_id_to_piece:
                self.id_to_ord.append(self.tokenizer.piece_to_ord(self.unspecial_id_to_piece[i]))
            else:
                break
            i += 1

        return self.id_to_ord


    # Copy vocabulary from model

    def get_id_to_piece_list(self):

        if self.id_to_piece is not None: return self.id_to_piece
        id_to_ord = self.get_id_to_ord_list()

        self.id_to_piece = [""] * self.tokenizer.vocab_size()
        for idx, p in self.tokenizer.enumerate_tokens():
            # if id_to_ord[idx] != -1:
            #     self.id_to_piece[idx] = chr(id_to_ord[idx])
            # else:
            #     self.id_to_piece[idx] = self.tokenizer.clean_special_chars(p)
            self.id_to_piece[idx] = p

        i = self.tokenizer.vocab_size()
        while True:
            if i in self.extended_id_to_piece:
                self.id_to_piece.append(self.extended_id_to_piece[i])
            elif i in self.unspecial_id_to_piece:
                self.id_to_piece.append(self.unspecial_id_to_piece[i])
            else:
                break
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
