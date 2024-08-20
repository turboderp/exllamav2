from __future__ import annotations

import torch
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
from exllamav2.tensor_p import BROADCAST_KV

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2.model import ExLlamaV2
    from exllamav2 import ExLlamaV2Tokenizer

class ExLlamaV2CacheBase:

    model = None
    max_seq_len: int
    batch_size: int

    current_seq_len: int

    key_states: list[torch.Tensor | None]
    value_states: list[torch.Tensor | None]
    key_scales: list[torch.Tensor | None]
    value_scales: list[torch.Tensor | None]
    num_key_value_heads: int
    num_hidden_layers: int
    head_dim: int

    dtype: torch.dtype
    weights_per_element_k: int
    weights_per_element_v: int
    has_scales: bool
    fixed_device: int | None


    def __init__(
        self,
        model: ExLlamaV2,
        batch_size: int,
        max_seq_len: int,
        dtype: torch.dtype,
        weights_per_element_k: int,
        weights_per_element_v: int,
        has_scales: bool,
        num_key_value_heads: int | None = None,
        fixed_device: int | None = None
    ):

        self.model = model
        self.max_seq_len = max_seq_len if max_seq_len != -1 else self.model.config.max_seq_len
        self.batch_size = batch_size
        self.dtype = dtype
        self.weights_per_element_k = weights_per_element_k
        self.weights_per_element_v = weights_per_element_v
        self.has_scales = has_scales

        self.key_states = []
        self.value_states = []
        self.key_scales = []
        self.value_scales = []

        self.num_key_value_heads = num_key_value_heads or self.model.config.num_key_value_heads
        self.num_hidden_layers = self.model.config.num_hidden_layers
        self.head_dim = self.model.config.head_dim

        self.current_seq_len = 0
        self.shape_basic = (self.batch_size, self.max_seq_len, self.num_key_value_heads, self.head_dim)
        self.shape_wk = (self.batch_size, self.max_seq_len, self.num_key_value_heads, self.head_dim // self.weights_per_element_k)
        self.shape_wv = (self.batch_size, self.max_seq_len, self.num_key_value_heads, self.head_dim // self.weights_per_element_v)
        self.shape_s = (self.batch_size, self.max_seq_len, self.num_key_value_heads, self.head_dim // 32)

        self.q_block = 0
        self.fixed_device = fixed_device


    def create_state_tensors(self,
                             copy_from: ExLlamaV2CacheBase | None,
                             lazy = False):

        assert copy_from is None or lazy == False, "Cannot use lazy cache initialization while copying"

        if copy_from:
            self.current_seq_len = copy_from.current_seq_len

        if not lazy:

            for i in range(self.num_hidden_layers):

                if copy_from is None:
                    device = self.model.cache_map.get(i, self.fixed_device)
                    p_key_states = torch.zeros(self.shape_wk, dtype = self.dtype, device = device).contiguous()
                    p_value_states = torch.zeros(self.shape_wv, dtype = self.dtype, device = device).contiguous()
                    if self.has_scales:
                        p_key_scales = torch.zeros(self.shape_s, dtype = torch.float16, device = device).contiguous()
                        p_value_scales = torch.zeros(self.shape_s, dtype = torch.float16, device = device).contiguous()
                else:
                    p_key_states = copy_from.key_states[i].clone()
                    p_value_states = copy_from.value_states[i].clone()
                    if self.has_scales:
                        p_key_scales = copy_from.key_scales[i].clone()
                        p_value_scales = copy_from.value_scales[i].clone()

                self.key_states.append(p_key_states)
                self.value_states.append(p_value_states)
                if self.has_scales:
                    self.key_scales.append(p_key_scales)
                    self.value_scales.append(p_value_scales)

        else:

            for i in range(self.num_hidden_layers):

                self.key_states.append(None)
                self.value_states.append(None)
                if self.has_scales:
                    self.key_scales.append(None)
                    self.value_scales.append(None)


    def update_cache_tensors(self):

        for k, v in self.model.cache_map.items():

            self.touch_device(v)

            if self.key_states[k] is not None:

                if str(self.key_states[k].device) == v: continue
                self.key_states[k] = None
                self.value_states[k] = None

            p_key_states = torch.zeros(self.shape_wk, dtype = self.dtype, device = v).contiguous()
            p_value_states = torch.zeros(self.shape_wv, dtype = self.dtype, device = v).contiguous()
            self.key_states[k] = p_key_states
            self.value_states[k] = p_value_states
            if self.has_scales:
                p_key_scales = torch.zeros(self.shape_s, dtype = torch.float16, device = v).contiguous()
                p_value_scales = torch.zeros(self.shape_s, dtype = torch.float16, device = v).contiguous()
                self.key_scales[k] = p_key_scales
                self.value_scales[k] = p_value_scales


    def roll_left(self):

        for i in range(self.model.config.num_hidden_layers):

            self.key_states[i] = torch.roll(self.key_states[i], shifts = -1, dims = 2)
            self.value_states[i] = torch.roll(self.value_states[i], shifts = -1, dims = 2)

        self.current_seq_len -= 1


    def get_kv_state(self,
                     layer_idx: int,
                     batch_size: int,
                     offset: int,
                     width: int,
                     page_size: int = 0,
                     cache_seqlens: torch.Tensor | None = None,
                     block_table: torch.Tensor | None = None) -> (torch.Tensor, torch.Tensor):
        raise NotImplementedError


    def store_kv_state(self,
                       layer_idx: int,
                       batch_size: int,
                       offset: int,
                       width: int,
                       page_size: int = 0,
                       cache_seqlens: torch.Tensor | None = None,
                       block_table: torch.Tensor | None = None):
        raise NotImplementedError


    @torch.inference_mode
    def copy_states(self,
                    target: ExLlamaV2CacheBase,
                    from_column: int,
                    from_columns: int,
                    to_column: int,
                    to_columns: int,
                    from_row: int,
                    from_rows: int,
                    to_row: int,
                    to_rows: int):

        assert from_rows == 1
        assert from_columns == to_columns
        assert to_column + to_columns <= target.max_seq_len
        assert from_column + from_columns <= self.max_seq_len

        num_hidden_layers = self.model.config.num_hidden_layers

        tensors = [
            (self.key_states, target.key_states),
            (self.value_states, target.value_states),
        ]
        if self.has_scales:
            tensors += [
                (self.key_scales, target.key_scales),
                (self.value_scales, target.value_scales),
            ]

        for i in range(num_hidden_layers):
            for (src, dst) in tensors:
                src_view = src[i].narrow(0, from_row, from_rows).narrow(1, from_column, from_columns)
                dst_view = dst[i].narrow(0, to_row, to_rows).narrow(1, to_column, to_columns)
                if to_rows > 1:
                    src_view = src_view.expand_as(dst_view)
                dst_view.copy_(src_view, non_blocking = True)


    def touch_device(self, device):
        pass


    def all_tensors(self):
        raise NotImplementedError()


    def reset(self):
        self.current_seq_len = 0


class ExLlamaV2Cache(ExLlamaV2CacheBase):
    """
    FP16 cache
    """

    def __init__(
        self,
        model: ExLlamaV2,
        batch_size: int = 1,
        max_seq_len: int = -1,
        copy_from: ExLlamaV2Cache | None = None,
        lazy: bool = False,
        num_key_value_heads: int | None = None,
        fixed_device: int | None = None
    ):

        super().__init__(
            model,
            batch_size,
            max_seq_len,
            torch.half,
            1, 1,
            False,
            num_key_value_heads,
            fixed_device
        )

        self.create_state_tensors(copy_from, lazy)


    def get_kv_state(self,
                     layer_idx: int,
                     batch_size: int,
                     offset: int,
                     width: int,
                     page_size: int = 0,
                     cache_seqlens: torch.Tensor | None = None,
                     block_table: torch.Tensor | None = None) -> (torch.Tensor, torch.Tensor):

        return self.key_states[layer_idx], self.value_states[layer_idx]


    def store_kv_state(self,
                       layer_idx: int,
                       batch_size: int,
                       offset: int,
                       width: int,
                       page_size: int = 0,
                       cache_seqlens: torch.Tensor | None = None,
                       block_table: torch.Tensor | None = None):

        pass


    def footprint(self):

        fp = []
        for layer in self.key_states + self.value_states:
            dev = layer.device.index
            while len(fp) <= dev: fp.append(0)
            fp[dev] += layer.numel() * 2
        return fp


    def clone(self):

        new = ExLlamaV2Cache(self.model, self.batch_size, self.max_seq_len, self)
        return new

    def all_tensors(self):
        return self.key_states + self.value_states


class ExLlamaV2Cache_8bit(ExLlamaV2CacheBase):
    """
    8-bit cache. Keys and values are compressed to FP8 (e5m2) format by truncation.
    """

    def __init__(
        self,
        model: ExLlamaV2,
        batch_size: int = 1,
        max_seq_len: int = -1,
        copy_from: ExLlamaV2Cache_8bit | None = None,
        lazy: bool = False,
        num_key_value_heads: int | None = None,
        fixed_device: int | None = None
    ):

        super().__init__(
            model,
            batch_size,
            max_seq_len,
            torch.uint8,
            1, 1,
            False,
            num_key_value_heads,
            fixed_device
        )

        self.create_state_tensors(copy_from, lazy)

        # Create temp FP16 tensors for accessing FP8 layers

        self.temp_tensors = {}
        if not lazy:
            devs = self.model.get_cache_devices() if self.fixed_device is None else [self.fixed_device]
            for device in devs: self.touch_device(device)


    def touch_device(self, device):

        if device in self.temp_tensors: return
        k = torch.zeros(self.shape_basic, dtype = torch.float16, device = device).contiguous()
        v = torch.zeros(self.shape_basic, dtype = torch.float16, device = device).contiguous()
        self.temp_tensors[device] = (k, v)


    def get_kv_state(self,
                     layer_idx: int,
                     batch_size: int,
                     offset: int,
                     width: int,
                     page_size: int = 0,
                     cache_seqlens: torch.Tensor | None = None,
                     block_table: torch.Tensor | None = None) -> (torch.Tensor, torch.Tensor):

        device = self.model.cache_map.get(layer_idx, self.fixed_device)

        temp_key_state, temp_value_state = self.temp_tensors[device]
        if width > 0: ext_c.fp8_to_fp16(self.key_states[layer_idx], temp_key_state, batch_size, offset, width)
        if width > 0: ext_c.fp8_to_fp16(self.value_states[layer_idx], temp_value_state, batch_size, offset, width)
        return temp_key_state, temp_value_state


    def store_kv_state(self,
                       layer_idx: int,
                       batch_size: int,
                       offset: int,
                       width: int,
                       page_size: int = 0,
                       cache_seqlens: torch.Tensor | None = None,
                       block_table: torch.Tensor | None = None):

        device = self.model.cache_map.get(layer_idx, self.fixed_device)

        temp_key_state, temp_value_state = self.temp_tensors[device]
        if width > 0: ext_c.fp16_to_fp8(temp_key_state, self.key_states[layer_idx], batch_size, offset, width)
        if width > 0: ext_c.fp16_to_fp8(temp_value_state, self.value_states[layer_idx], batch_size, offset, width)


    def footprint(self) -> list[int]:

        fp = []
        for layer in self.key_states + self.value_states:
            dev = layer.device.index
            while len(fp) <= dev: fp.append(0)
            fp[dev] += layer.numel() * 1
        for temp_k, temp_v in self.temp_tensors.values():
            fp[temp_k.device.index] += temp_k.numel() * 2
            fp[temp_v.device.index] += temp_v.numel() * 2
        return fp


    def clone(self) -> ExLlamaV2Cache_8bit:

        new = ExLlamaV2Cache_8bit(self.model, self.batch_size, self.max_seq_len, self)
        return new

    def all_tensors(self):
        return self.key_states + self.value_states


class ExLlamaV2Cache_Q(ExLlamaV2CacheBase):
    """
    Q cache. Uses grouped RTN quantization for keys/values
    """

    wbits: int

    def __init__(
        self,
        model: ExLlamaV2,
        batch_size: int = 1,
        max_seq_len: int = -1,
        copy_from: ExLlamaV2Cache_Q4 | None = None,
        lazy: bool = False,
        weights_per_byte_k: int = -1,
        weights_per_byte_v: int = -1,
        num_key_value_heads: int | None = None,
        fixed_device: int | None = None
    ):

        super().__init__(
            model,
            batch_size,
            max_seq_len,
            torch.uint8,
            weights_per_byte_k,
            weights_per_byte_v,
            True,
            num_key_value_heads,
            fixed_device
        )
        cfg = self.model.config

        self.create_state_tensors(copy_from, lazy)

        # Models with odd key/value dims need to quantize/dequantize in multi-token blocks. Make sure the quant
        # blocksize aligns with a whole number of tokens

        if not num_key_value_heads:
            num_key_value_heads = cfg.num_key_value_heads

        Q_CACHE_BLOCKSIZE_Q = 512
        kv_dim = num_key_value_heads * cfg.head_dim
        self.q_block = 1
        while (kv_dim * self.q_block) % Q_CACHE_BLOCKSIZE_Q:
            self.q_block += 1
        self.max_seq_len = (self.max_seq_len + self.q_block - 1) // self.q_block * self.q_block

        # Create temp FP16 tensors for accessing Q4 layers

        self.temp_tensors = {}
        if not lazy:
            devs = self.model.get_cache_devices() if self.fixed_device is None else [self.fixed_device]
            for device in devs: self.touch_device(device)

        # Calibration mode

        self.calibrated = False
        self.calibrating = False
        self.calibration_rows = [0] * cfg.num_hidden_layers
        self.calibration_k = {}
        self.calibration_v = {}


    def touch_device(self, device):

        if device in self.temp_tensors: return
        k = torch.zeros(self.shape_basic, dtype = torch.float16, device = device).contiguous()
        v = torch.zeros(self.shape_basic, dtype = torch.float16, device = device).contiguous()
        self.temp_tensors[device] = (k, v)


    def get_kv_state(self,
                     layer_idx: int,
                     batch_size: int,
                     offset: int,
                     width: int,
                     page_size: int = 0,
                     cache_seqlens: torch.Tensor | None = None,
                     block_table: torch.Tensor | None = None) -> (torch.Tensor, torch.Tensor):

        device = self.model.cache_map.get(layer_idx, self.fixed_device)

        temp_key_state, temp_value_state = self.temp_tensors[device]
        if width == 0: return temp_key_state, temp_value_state

        if self.q_block > 1 and not page_size:
            a = offset
            b = offset + width
            a = a // self.q_block * self.q_block
            b = (b + self.q_block - 1) // self.q_block * self.q_block
            offset = a
            width = b - a

        ext_c.q_to_fp16_kv(
            self.key_states[layer_idx],
            temp_key_state,
            self.key_scales[layer_idx],
            self.value_states[layer_idx],
            temp_value_state,
            self.value_scales[layer_idx],
            batch_size,
            offset,
            width,
            page_size,
            cache_seqlens if cache_seqlens is not None else none_tensor,
            block_table if block_table is not None else none_tensor,
            # none_tensor,
            # none_tensor
            self.calibration_k[layer_idx] if self.calibrated else none_tensor,
            self.calibration_v[layer_idx] if self.calibrated else none_tensor,
            self.wbits
        )

        # if self.calibrated:
        #     temp_key_state *= self.calibration_k[layer_idx]
        #     temp_value_state *= self.calibration_v[layer_idx]

        return temp_key_state, temp_value_state


    def store_kv_state(self,
                       layer_idx: int,
                       batch_size: int,
                       offset: int,
                       width: int,
                       page_size: int = 0,
                       cache_seqlens: torch.Tensor | None = None,
                       block_table: torch.Tensor | None = None):

        if width == 0: return

        if self.q_block > 1 and not page_size:
            a = offset
            b = offset + width
            a = a // self.q_block * self.q_block
            b = (b + self.q_block - 1) // self.q_block * self.q_block
            offset = a
            width = b - a

        device = self.model.cache_map.get(layer_idx, self.fixed_device)
        temp_key_state, temp_value_state = self.temp_tensors[device]

        # if self.calibrated:
        #     temp_key_state /= self.calibration_k[layer_idx]
        #     temp_value_state /= self.calibration_v[layer_idx]

        ext_c.fp16_to_q_kv(
            temp_key_state,
            self.key_states[layer_idx],
            self.key_scales[layer_idx],
            temp_value_state,
            self.value_states[layer_idx],
            self.value_scales[layer_idx],
            batch_size,
            offset,
            width,
            page_size,
            cache_seqlens if cache_seqlens is not None else none_tensor,
            block_table if block_table is not None else none_tensor,
            # none_tensor,
            # none_tensor
            self.calibration_k[layer_idx] if self.calibrated else none_tensor,
            self.calibration_v[layer_idx] if self.calibrated else none_tensor,
            self.wbits
        )

        # Collect calibration data

        if self.calibrating:

            cfg = self.model.config

            if layer_idx not in self.calibration_k:
                self.calibration_k[layer_idx] = torch.zeros(
                    (cfg.num_key_value_heads, cfg.head_dim,),
                    dtype = torch.float,
                    device = temp_key_state.device
                )
                self.calibration_v[layer_idx] = torch.zeros(
                    (cfg.num_key_value_heads, cfg.head_dim,),
                    dtype = torch.float,
                    device = temp_key_state.device
                )

            b, l, h, d = temp_key_state.shape
            cal_k = self.calibration_k[layer_idx]
            cal_v = self.calibration_v[layer_idx]
            cal_k_input = temp_key_state[:, offset:offset+width, :, :].view(b * width, h * d)
            cal_v_input = temp_value_state[:, offset:offset+width, :, :].view(b * width, h * d)
            cal_k_sum = torch.norm(cal_k_input, p = 1, dim = 0, dtype = torch.float)
            cal_v_sum = torch.norm(cal_v_input, p = 1, dim = 0, dtype = torch.float)
            cal_k.add_(cal_k_sum.view(h, d))
            cal_v.add_(cal_v_sum.view(h, d))
            self.calibration_rows[layer_idx] += width


    def footprint(self) -> list[int]:

        fp = []
        for layer in self.key_states + self.value_states:
            dev = layer.device.index
            while len(fp) <= dev: fp.append(0)
            fp[dev] += layer.numel() * 1
        for layer in self.key_scales + self.value_scales:
            dev = layer.device.index
            while len(fp) <= dev: fp.append(0)
            fp[dev] += layer.numel() * 2
        for temp_k, temp_v in self.temp_tensors.values():
            fp[temp_k.device.index] += temp_k.numel() * 2
            fp[temp_v.device.index] += temp_v.numel() * 2
        return fp


    def clone(self) -> ExLlamaV2Cache_Q4:

        new = ExLlamaV2Cache_Q4(self.model, self.batch_size, self.max_seq_len, self)
        return new


    def all_tensors(self):
        return self.key_states + self.value_states + self.key_scales + self.value_scales


    def calibrate(self,
        tokenizer: ExLlamaV2Tokenizer,
        num_batches = 8,
        num_samples_per_batch = 256
    ):
        """
        Unfinished
        """

        assert self.max_seq_len >= num_samples_per_batch, \
            f"Cache max_seq_len must be at least {num_samples_per_batch} to calibrate."

        self.calibrating = True
        torch.manual_seed(123)

        for _ in range(num_batches):

            input_ids = torch.randint(
                low = 0,
                high = tokenizer.get_vocab_size() - 1,
                size = (1, num_samples_per_batch),
                dtype = torch.long
            )

            self.reset()
            self.model.forward(input_ids, preprocess_only = True, cache = self)

        self.calibrating = False

        for i in range(self.model.config.num_hidden_layers):
            cal_k = self.calibration_k[i] / self.calibration_rows[i]  # self.calibration_k[i].mean()
            cal_v = self.calibration_v[i] / self.calibration_rows[i]  # self.calibration_v[i].mean()
            cal_k = cal_k ** (1/8)
            cal_v = cal_v ** (1/8)
            cal_k = cal_k.half() * (-1)
            cal_v = cal_v.half() * (-1)
            self.calibration_k[i] = cal_k
            self.calibration_v[i] = cal_v
        self.calibrating = False
        # self.calibrated = True


class ExLlamaV2Cache_Q4(ExLlamaV2Cache_Q):

    def __init__(
        self,
        model: ExLlamaV2,
        batch_size: int = 1,
        max_seq_len: int = -1,
        copy_from: ExLlamaV2Cache_Q4 | None = None,
        lazy: bool = False,
        num_key_value_heads: int | None = None,
        fixed_device: int | None = None
    ):

        super().__init__(
            model,
            batch_size,
            max_seq_len,
            copy_from,
            lazy,
            2, 2,
            num_key_value_heads,
            fixed_device
        )
        self.wbits = 4


class ExLlamaV2Cache_Q6(ExLlamaV2Cache_Q):

    def __init__(
        self,
        model: ExLlamaV2,
        batch_size: int = 1,
        max_seq_len: int = -1,
        copy_from: ExLlamaV2Cache_Q6 | None = None,
        lazy: bool = False,
        num_key_value_heads: int | None = None,
        fixed_device: int | None = None
    ):

        super().__init__(
            model,
            batch_size,
            max_seq_len,
            copy_from,
            lazy,
            1, 2,
            num_key_value_heads,
            fixed_device
        )
        self.wbits = 6


class ExLlamaV2Cache_Q8(ExLlamaV2Cache_Q):

    def __init__(
        self,
        model: ExLlamaV2,
        batch_size: int = 1,
        max_seq_len: int = -1,
        copy_from: ExLlamaV2Cache_Q8 | None = None,
        lazy: bool = False,
        num_key_value_heads: int | None = None,
        fixed_device: int | None = None
    ):

        super().__init__(
            model,
            batch_size,
            max_seq_len,
            copy_from,
            lazy,
            1, 1,
            num_key_value_heads,
            fixed_device
        )
        self.wbits = 8


class ExLlamaV2Cache_TP(ExLlamaV2CacheBase):

    caches: list[ExLlamaV2CacheBase]

    max_seq_len: int
    batch_size: int

    current_seq_len: int

    def __init__(
        self,
        model: ExLlamaV2,
        base: type = ExLlamaV2Cache,
        batch_size: int = 1,
        max_seq_len: int = -1
    ):
        super().__init__(model, batch_size, max_seq_len, torch.half, 1, 1, False, None)

        assert model.tp_context is not None, \
            "Cannot create TP cache unless model is loaded with load_tp()"

        self.caches = [
            base(
                model = model,
                batch_size = batch_size,
                max_seq_len = max_seq_len,
                copy_from = None,
                lazy = False,
                num_key_value_heads = b - a,
                fixed_device = idx
            )
            if b - a > 0 else None
            for idx, a, b in model.tp_context.get_split(BROADCAST_KV)
        ]

        # for idx, cache in enumerate(self.caches):
        #     if cache is None: continue
        #     cache.fixed_device = idx
        #     cache.create_state_tensors(copy_from = None, lazy = False)


    def roll_left(self):
        for cache in self.caches:
            cache.roll_left()


    def get_kv_state(
        self,
        layer_idx: int,
        batch_size: int,
        offset: int,
        width: int,
        page_size: int = 0,
        cache_seqlens: torch.Tensor | list[torch.Tensor] | None = None,
        block_table: torch.Tensor | list[torch.Tensor] | None = None
    ) -> (torch.Tensor, torch.Tensor):
        kc, vc = [], []
        for idx, cache in enumerate(self.caches):
            k, v = cache.get_kv_state(
                layer_idx,
                batch_size,
                offset,
                width,
                page_size,
                cache_seqlens[idx] if cache_seqlens else None,
                block_table[idx] if block_table else None
            )
            kc.append(k)
            vc.append(v)
        return kc, vc


    def store_kv_state(
        self,
        layer_idx: int,
        batch_size: int,
        offset: int,
        width: int,
        page_size: int = 0,
        cache_seqlens: torch.Tensor | list[torch.Tensor] | None = None,
        block_table: torch.Tensor | list[torch.Tensor] | None = None
    ):
        for idx, cache in enumerate(self.caches):
            cache.store_kv_state(
                layer_idx,
                batch_size,
                offset,
                width,
                page_size,
                cache_seqlens[idx] if cache_seqlens else None,
                block_table[idx] if block_table else None
            )


    @torch.inference_mode
    def copy_states(
        self,
        target: ExLlamaV2Cache_TP,
        from_column: int,
        from_columns: int,
        to_column: int,
        to_columns: int,
        from_row: int,
        from_rows: int,
        to_row: int,
        to_rows: int
    ):
        for cache, tcache in zip(self.caches, target.caches):
            cache.copy_states(
                tcache,
                from_column,
                from_columns,
                to_column,
                to_columns,
                from_row,
                from_rows,
                to_row,
                to_rows
            )


    def touch_device(self, device):
        raise NotImplementedError()


    def all_tensors(self):
        tensors = []
        for cache in self.caches:
            tensors += cache.all_tensors()
        return tensors


    def reset(self):
        self.current_seq_len = 0
