
# Common keys

layer_keys_llama_norms = [["input_layernorm"],
                          ["post_attention_layernorm"]]
layer_keys_cohere_norms = [["input_layernorm"]]
layer_keys_yi_norms = [["ln1", "input_layernorm"],
                       ["ln2", "post_attention_layernorm"]]
layer_keys_llama_attn = [["self_attn.q_proj"],
                         ["self_attn.k_proj"],
                         ["self_attn.v_proj"],
                         ["self_attn.o_proj"]]
layer_keys_llama_mlp = [["mlp.down_proj"],
                        ["mlp.gate_proj"],
                        ["mlp.up_proj"]]
layer_keys_llama_mlp_swiglu = [["mlp.swiglu.w12"],
                               ["mlp.swiglu.w3"]]
layer_keys_starcoder2_mlp = [["mlp.c_fc"],
                             ["mlp.c_proj"]]
expect_keys_llama = [["lm_head"],
                     ["model.norm"],
                     ["model.embed_tokens"]]
expect_keys_gemma = [["model.norm"],
                     ["model.embed_tokens"]]
expect_keys_starcoder2 = [["model.norm"],
                          ["model.embed_tokens"]]

class ExLlamaV2ArchParams:

    def __init__(self, arch_string, read_config):

        self.arch_string = arch_string
        arch_recognized = False

        self.expect_keys = []  # Keys to expect in model dict
        self.layer_keys = []  # Keys to expect in model dict, per layer

        self.fused_mlp_key_12 = None
        self.fused_mlp_key_3 = None

        # Mistral

        if arch_string == "MistralForCausalLM":
            arch_recognized = True
            self.layer_keys += \
                layer_keys_llama_norms + \
                layer_keys_llama_attn + \
                layer_keys_llama_mlp
            self.expect_keys += \
                expect_keys_llama
            self.norm_eps_key = "rms_norm_eps"
            self.attention_bias_qkv = False
            self.attention_bias_o = False
            self.mlp_bias = False
            self.mlp_gate = True
            self.mlp_key_gate = ".mlp.gate_proj"
            self.mlp_key_up = ".mlp.up_proj"
            self.mlp_key_down = ".mlp.down_proj"
            self.mlp_act_func = "silu"
            self.is_moe = False
            self.norm = "rmsnorm"
            self.lm_head_key = "lm_head"
            self.normalize_embeddings = False
            self.norm_key_1 = ".input_layernorm"
            self.norm_key_2 = ".post_attention_layernorm"
            self.norm_constant_bias = 0
            self.parallel_decoder_blocks = False
            self.requires_bos = False
            self.rope_neox_style = True

        # Mixtral

        if arch_string == "MixtralForCausalLM":
            arch_recognized = True
            self.layer_keys += \
                layer_keys_llama_norms + \
                layer_keys_llama_attn + \
                [[f"block_sparse_moe.experts.{e}.w{w}" for e in range(8) for w in range(3)]] + \
                [["block_sparse_moe.gate"]]
            self.expect_keys += \
                expect_keys_llama
            self.norm_eps_key = "rms_norm_eps"
            self.attention_bias_qkv = False
            self.attention_bias_o = False
            self.mlp_bias = False
            self.mlp_gate = True
            self.mlp_key_gate = ".block_sparse_moe.experts.*.w1"
            self.mlp_key_up = ".block_sparse_moe.experts.*.w3"
            self.mlp_key_down = ".block_sparse_moe.experts.*.w2"
            self.mlp_key_expert_gate = ".block_sparse_moe.gate"
            self.mlp_act_func = "silu"
            self.is_moe = True
            self.norm = "rmsnorm"
            self.lm_head_key = "lm_head"
            self.normalize_embeddings = False
            self.norm_key_1 = ".input_layernorm"
            self.norm_key_2 = ".post_attention_layernorm"
            self.norm_constant_bias = 0
            self.parallel_decoder_blocks = False
            self.requires_bos = False
            self.rope_neox_style = True

        # Yi

        if arch_string == "YiForCausalLM":
            arch_recognized = True
            self.layer_keys += \
                layer_keys_yi_norms + \
                layer_keys_llama_attn + \
                layer_keys_llama_mlp
            self.expect_keys += \
                expect_keys_llama
            self.norm_eps_key = "rms_norm_eps"
            self.attention_bias_qkv = False
            self.attention_bias_o = False
            self.mlp_bias = False
            self.mlp_gate = True
            self.mlp_key_gate = ".mlp.gate_proj"
            self.mlp_key_up = ".mlp.up_proj"
            self.mlp_key_down = ".mlp.down_proj"
            self.mlp_act_func = "silu"
            self.is_moe = False
            self.norm = "rmsnorm"
            self.lm_head_key = "lm_head"
            self.normalize_embeddings = False
            self.norm_key_1 = ".ln1"
            self.norm_key_2 = ".ln2"
            self.norm_constant_bias = 0
            self.parallel_decoder_blocks = False
            self.requires_bos = False
            self.rope_neox_style = True

        # Orion

        if arch_string == "OrionForCausalLM":
            arch_recognized = True
            self.layer_keys += \
                layer_keys_llama_norms + \
                layer_keys_llama_attn + \
                layer_keys_llama_mlp
            self.expect_keys += \
                expect_keys_llama
            self.norm_eps_key = "rms_norm_eps"
            self.attention_bias_qkv = False
            self.attention_bias_o = False
            self.mlp_bias = False
            self.mlp_gate = True
            self.mlp_key_gate = ".mlp.gate_proj"
            self.mlp_key_up = ".mlp.up_proj"
            self.mlp_key_down = ".mlp.down_proj"
            self.mlp_act_func = "silu"
            self.is_moe = False
            self.norm = "layernorm"
            self.lm_head_key = "lm_head"
            self.normalize_embeddings = False
            self.norm_key_1 = ".input_layernorm"
            self.norm_key_2 = ".post_attention_layernorm"
            self.norm_constant_bias = 0
            self.parallel_decoder_blocks = False
            self.requires_bos = False
            self.rope_neox_style = True

        # Qwen2 (1.5)

        if arch_string == "Qwen2ForCausalLM":
            arch_recognized = True
            self.layer_keys += \
                layer_keys_llama_norms + \
                layer_keys_llama_attn + \
                layer_keys_llama_mlp
            self.expect_keys += \
                expect_keys_llama
            self.norm_eps_key = "rms_norm_eps"
            self.attention_bias_qkv = True
            self.attention_bias_o = False
            self.mlp_bias = False
            self.mlp_gate = True
            self.mlp_key_gate = ".mlp.gate_proj"
            self.mlp_key_up = ".mlp.up_proj"
            self.mlp_key_down = ".mlp.down_proj"
            self.mlp_act_func = "silu"
            self.is_moe = False
            self.norm = "rmsnorm"
            self.lm_head_key = "lm_head"
            self.normalize_embeddings = False
            self.norm_key_1 = ".input_layernorm"
            self.norm_key_2 = ".post_attention_layernorm"
            self.norm_constant_bias = 0
            self.parallel_decoder_blocks = False
            self.requires_bos = False
            self.rope_neox_style = True

        # Gemma

        if arch_string == "GemmaForCausalLM":
            arch_recognized = True
            self.layer_keys += \
                layer_keys_llama_norms + \
                layer_keys_llama_attn + \
                layer_keys_llama_mlp
            self.expect_keys += \
                expect_keys_gemma
            self.norm_eps_key = "rms_norm_eps"
            self.attention_bias_qkv = False
            self.attention_bias_o = False
            self.mlp_bias = False
            self.mlp_gate = True
            self.mlp_key_gate = ".mlp.gate_proj"
            self.mlp_key_up = ".mlp.up_proj"
            self.mlp_key_down = ".mlp.down_proj"
            self.mlp_act_func = "gelu"
            self.is_moe = False
            self.norm = "rmsnorm"
            self.lm_head_key = "model.embed_tokens"
            self.normalize_embeddings = True
            self.norm_key_1 = ".input_layernorm"
            self.norm_key_2 = ".post_attention_layernorm"
            self.norm_constant_bias = 1
            self.parallel_decoder_blocks = False
            self.requires_bos = True
            self.rope_neox_style = True

        # StarCoder2

        if arch_string == "Starcoder2ForCausalLM":
            arch_recognized = True
            self.layer_keys += \
                layer_keys_llama_norms + \
                layer_keys_llama_attn + \
                layer_keys_starcoder2_mlp
            self.expect_keys += \
                expect_keys_starcoder2
            self.norm_eps_key = "norm_epsilon"
            self.attention_bias_qkv = True
            self.attention_bias_o = True
            self.mlp_bias = True
            self.mlp_gate = False
            self.mlp_key_up = ".mlp.c_fc"
            self.mlp_key_down = ".mlp.c_proj"
            self.mlp_act_func = "gelu"
            self.is_moe = False
            self.norm = "layernorm"
            self.lm_head_key = "model.embed_tokens"
            self.normalize_embeddings = False
            self.norm_key_1 = ".input_layernorm"
            self.norm_key_2 = ".post_attention_layernorm"
            self.norm_constant_bias = 0
            self.parallel_decoder_blocks = False
            self.requires_bos = False
            self.rope_neox_style = True

        # GemMoE

        if arch_string == "GemmoeForCausalLM":
            arch_recognized = True
            print(f" !! Warning, Gemmoe support is experimental and has not been fully tested")
            self.layer_keys += \
                layer_keys_llama_norms + \
                layer_keys_llama_attn + \
                [[f"block_sparse_moe.experts.{e}.w{w}" for e in range(8) for w in range(3)]] + \
                [["block_sparse_moe.gate"]]
            self.expect_keys += \
                expect_keys_gemma
            self.norm_eps_key = "rms_norm_eps"
            self.attention_bias_qkv = False
            self.attention_bias_o = False
            self.mlp_bias = False
            self.mlp_gate = True
            self.mlp_key_gate = ".block_sparse_moe.experts.*.w1"
            self.mlp_key_up = ".block_sparse_moe.experts.*.w3"
            self.mlp_key_down = ".block_sparse_moe.experts.*.w2"
            self.mlp_key_expert_gate = ".block_sparse_moe.gate"
            self.mlp_act_func = "gelu"
            self.is_moe = True
            self.norm = "rmsnorm"
            self.lm_head_key = "model.embed_tokens"
            self.normalize_embeddings = True
            self.norm_key_1 = ".input_layernorm"
            self.norm_key_2 = ".post_attention_layernorm"
            self.norm_constant_bias = 1
            self.parallel_decoder_blocks = False
            self.requires_bos = True
            self.rope_neox_style = True

        # Cohere

        if arch_string == "CohereForCausalLM":
            arch_recognized = True
            self.layer_keys += \
                layer_keys_cohere_norms + \
                layer_keys_llama_attn + \
                layer_keys_llama_mlp
            self.expect_keys += \
                expect_keys_gemma
            self.norm_eps_key = "layer_norm_eps"
            self.attention_bias_qkv = False
            self.attention_bias_o = False
            self.mlp_bias = False
            self.mlp_gate = True
            self.mlp_key_gate = ".mlp.gate_proj"
            self.mlp_key_up = ".mlp.up_proj"
            self.mlp_key_down = ".mlp.down_proj"
            self.mlp_act_func = "silu"
            self.is_moe = False
            self.norm = "layernorm"
            self.lm_head_key = "model.embed_tokens"
            self.normalize_embeddings = False
            self.norm_key_1 = ".input_layernorm"
            self.norm_key_2 = None
            self.norm_constant_bias = 0
            self.parallel_decoder_blocks = True
            self.requires_bos = True
            self.rope_neox_style = False

        # Llama (default + fallback)

        if arch_string != "LlamaForCausalLM" and not arch_recognized:
            print(f" !! Warning, unknown architecture: {arch_string}")
            print(f" !! Loading as LlamaForCausalLM")
            self.arch_string = "LlamaForCausalLM"
        if not arch_recognized:
            self.layer_keys += \
                layer_keys_llama_norms + \
                layer_keys_llama_attn + \
                layer_keys_llama_mlp
            self.expect_keys += \
                expect_keys_llama
            self.norm_eps_key = "rms_norm_eps"
            self.attention_bias_qkv = False
            self.attention_bias_o = False
            self.mlp_bias = False
            self.mlp_gate = True
            self.mlp_key_gate = ".mlp.gate_proj"
            self.mlp_key_up = ".mlp.up_proj"
            self.mlp_key_down = ".mlp.down_proj"
            self.mlp_act_func = "silu"
            self.is_moe = False
            self.norm = "rmsnorm"
            self.lm_head_key = "lm_head"
            self.normalize_embeddings = False
            self.norm_key_1 = ".input_layernorm"
            self.norm_key_2 = ".post_attention_layernorm"
            self.norm_constant_bias = 0
            self.parallel_decoder_blocks = False
            self.requires_bos = False
            self.rope_neox_style = True

        # Arch overrides

        if read_config.get("attention_bias", False):
            self.attention_bias_qkv = True
            self.attention_bias_o = True


    def make_fused_mlp(self):

        for x in layer_keys_llama_mlp: self.layer_keys.remove(x)
        self.layer_keys += layer_keys_llama_mlp_swiglu
        self.fused_mlp_key_12 = layer_keys_llama_mlp_swiglu[0][0]
        self.fused_mlp_key_3 = layer_keys_llama_mlp_swiglu[1][0]
