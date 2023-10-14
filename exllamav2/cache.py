import torch
from exllamav2.ext import exllamav2_ext as ext_c


class CacheBase:
    def __init__(self, batch_size: int, max_seq_len: int, num_key_value_heads: int):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_hidden_layers = num_key_value_heads
        self.num_key_value_heads = num_key_value_heads
        self.key_states = []
        self.value_states = []

    def get_key_state(self, layer_idx: int) -> torch.Tensor:
        return self.key_states[layer_idx]

    def get_value_state(self, layer_idx: int) -> torch.Tensor:
        return self.value_states[layer_idx]

    def store_tmp_key_state(self, layer_idx: int):
        pass

    def store_tmp_value_state(self, layer_idx: int):
        pass


class Cache16Bit(CacheBase):
    def __init__(self, model, batch_size: int, max_seq_len: int, num_key_value_heads: int, head_dim: int, copy_from = None):
        CacheBase.__init__(self, batch_size, max_seq_len, num_key_value_heads)

        self.model = model
        for i in range(self.num_hidden_layers):
            if copy_from is None:
                # create kv cache use uint8
                p_key_states = torch.zeros(self.batch_size, self.max_seq_len, num_key_value_heads, head_dim, dtype = torch.float16, device = self.model.cache_map[i])
                p_value_states = torch.zeros(self.batch_size, self.max_seq_len, num_key_value_heads, head_dim, dtype = torch.float16, device = self.model.cache_map[i])

            else:
                p_key_states = copy_from.key_states[i].clone()
                p_value_states = copy_from.value_states[i].clone()

            self.key_states.append(p_key_states)
            self.value_states.append(p_value_states)


class Cache8Bit(CacheBase):
    def __init__(self, model, batch_size: int, max_seq_len: int, num_key_value_heads: int, head_dim: int, copy_from = None):
        CacheBase.__init__(self, batch_size, max_seq_len, num_key_value_heads)
        self.model = model
        for i in range(self.num_hidden_layers):
            if copy_from is None:
                # create kv cache use uint8
                p_key_states = torch.zeros(self.batch_size, self.max_seq_len, num_key_value_heads, head_dim, dtype = torch.uint8, device = self.model.cache_map[i])
                p_value_states = torch.zeros(self.batch_size, self.max_seq_len, num_key_value_heads, head_dim, dtype = torch.uint8, device = self.model.cache_map[i])

            else:
                p_key_states = copy_from.key_states[i].clone()
                p_value_states = copy_from.value_states[i].clone()

            self.key_states.append(p_key_states)
            self.value_states.append(p_value_states)

        self.tmp_key_state = torch.zeros(self.batch_size, self.max_seq_len, num_key_value_heads, head_dim, dtype = torch.float16, device = self.model.cache_map[0])
        self.tmp_value_state = torch.zeros(self.batch_size, self.max_seq_len, num_key_value_heads, head_dim, dtype = torch.float16, device = self.model.cache_map[0])
        self.tensor_data_length = self.batch_size * self.max_seq_len * num_key_value_heads * head_dim

    def get_key_state(self, layer_idx: int) -> torch.Tensor:
        ext_c.array_fp8_to_fp16(self.key_states[layer_idx], self.tmp_key_state, self.tensor_data_length)
        return self.tmp_key_state

    def get_value_state(self, layer_idx: int) -> torch.Tensor:
        ext_c.array_fp8_to_fp16(self.value_states[layer_idx], self.tmp_value_state, self.tensor_data_length)
        return self.tmp_value_state

    def store_tmp_key_state(self, layer_idx: int):
        ext_c.array_fp16_to_fp8(self.tmp_key_state, self.key_states[layer_idx], self.tensor_data_length)

    def store_tmp_value_state(self, layer_idx: int):
        ext_c.array_fp16_to_fp8(self.tmp_value_state, self.value_states[layer_idx], self.tensor_data_length)


class ExLlamaV2Cache:

    def __init__(self, model, batch_size = 1, max_seq_len = -1, copy_from = None):

        self.model = model
        self.max_seq_len = max_seq_len if max_seq_len != -1 else self.model.config.max_seq_len
        self.batch_size = batch_size

        self.current_seq_len = 0

        # Preallocate full-length cache
        num_key_value_heads = self.model.config.num_key_value_heads
        head_dim = self.model.config.head_dim

        if self.model.config.kv_cache_mask is not None and self.model.config.kv_cache_mask == '8bit':
            self.cached = Cache8Bit(model, self.batch_size, self.max_seq_len, num_key_value_heads, head_dim, copy_from)
        else:
            self.cached = Cache16Bit(model, self.batch_size, self.max_seq_len, num_key_value_heads, head_dim, copy_from)

    def get_key_state(self, layer_idx: int) -> torch.Tensor:
        return self.cached.get_key_state(layer_idx)

    def get_value_state(self, layer_idx: int) -> torch.Tensor:
        return self.cached.get_value_state(layer_idx)

    def store_tmp_key_state(self, layer_idx: int):
        self.cached.store_tmp_key_state(layer_idx)

    def store_tmp_value_state(self, layer_idx: int):
        self.cached.store_tmp_value_state(layer_idx)

    def footprint(self):

        fp = []
        for layer in self.cached.key_states + self.cached.value_states:
            dev = layer.device.index
            while len(fp) <= dev: fp.append(0)
            fp[dev] += layer.numel() * 2

        return fp


    def clone(self):

        new = ExLlamaV2Cache(self.model, batch_size = self.batch_size, max_seq_len = self.max_seq_len, copy_from = self)
        return new


    def roll_left(self):

        for i in range(self.model.config.num_hidden_layers):

            self.cached.key_states[i] = torch.roll(self.cached.key_states[i], shifts = -1, dims = 2)
            self.cached.value_states[i] = torch.roll(self.cached.value_states[i], shifts = -1, dims = 2)

        self.current_seq_len -= 1


    def copy_states(self, target, from_column, from_columns, to_column, to_columns, from_row, from_rows, to_row, to_rows):

        assert from_rows == 1
        assert from_columns == to_columns
        assert to_column + to_columns <= target.max_seq_len
        assert from_column + from_columns <= self.max_seq_len

        num_hidden_layers = self.model.config.num_hidden_layers

        for i in range(num_hidden_layers):

            source_view_k = self.cached.key_states[i].narrow(0, from_row, from_rows).narrow(2, from_column, from_columns)
            source_view_v = self.cached.value_states[i].narrow(0, from_row, from_rows).narrow(2, from_column, from_columns)
            target_view_k = target.key_states[i].narrow(0, to_row, to_rows).narrow(2, to_column, to_columns)
            target_view_v = target.value_states[i].narrow(0, to_row, to_rows).narrow(2, to_column, to_columns)

            if to_rows > 1:

                source_view_k = source_view_k.expand_as(target_view_k)
                source_view_v = source_view_v.expand_as(target_view_v)

            target_view_k.copy_(source_view_k)
            target_view_v.copy_(source_view_v)

