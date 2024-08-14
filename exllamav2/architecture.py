from enum import IntEnum

# Common keys

layer_keys_llama_norms = [["input_layernorm"],
                          ["post_attention_layernorm"]]
layer_keys_cohere_norms = [["input_layernorm"]]
layer_keys_gpt2_norms = [["ln_1"],
                         ["ln_2"]]
layer_keys_yi_norms = [["ln1", "input_layernorm"],
                       ["ln2", "post_attention_layernorm"]]
layer_keys_gemma2_norms = [["input_layernorm"],
                           ["post_attention_layernorm"],
                           ["pre_feedforward_layernorm"],
                           ["post_feedforward_layernorm"]]
layer_keys_internlm2_norms = [["attention_norm"],
                              ["ffn_norm"]]
layer_keys_llama_attn = [["self_attn.q_proj"],
                         ["self_attn.k_proj"],
                         ["self_attn.v_proj"],
                         ["self_attn.o_proj"]]
layer_keys_gpt2_attn = [["self_attn.c_attn", "self_attn.q_proj"],
                        ["self_attn.c_attn", "self_attn.k_proj"],
                        ["self_attn.c_attn", "self_attn.v_proj"],
                        ["self_attn.o_proj"]]
layer_keys_internlm2_attn = [["self_attn.wqkv", "self_attn.q_proj"],
                             ["self_attn.wqkv", "self_attn.k_proj"],
                             ["self_attn.wqkv", "self_attn.v_proj"],
                             ["self_attn.o_proj"]]
layer_keys_dbrx_attn = [["self_attn.Wqkv", "self_attn.q_proj"],
                        ["self_attn.Wqkv", "self_attn.k_proj"],
                        ["self_attn.Wqkv", "self_attn.v_proj"],
                        ["self_attn.o_proj"]]
layer_keys_phi3_attn = [["self_attn.qkv_proj", "self_attn.q_proj"],
                        ["self_attn.qkv_proj", "self_attn.k_proj"],
                        ["self_attn.qkv_proj", "self_attn.v_proj"],
                        ["self_attn.o_proj"]]
layer_keys_llama_mlp = [["mlp.down_proj"],
                        ["mlp.gate_proj"],
                        ["mlp.up_proj"]]
layer_keys_internlm2_mlp = [["feed_forward.w1"],
                            ["feed_forward.w2"],
                            ["feed_forward.w3"]]
layer_keys_phi3_mlp = [["mlp.down_proj"],
                       ["mlp.gate_up_proj", "mlp.gate_proj"],
                       ["mlp.gate_up_proj", "mlp.up_proj"]]
layer_keys_mixtral_mlp = [["block_sparse_moe.experts.*.w1"],
                          ["block_sparse_moe.experts.*.w2"],
                          ["block_sparse_moe.experts.*.w3"],
                          ["block_sparse_moe.gate"]]
layer_keys_dbrx_mlp = [["block_sparse_moe.experts.*.v1", "block_sparse_moe.experts.v1"],
                       ["block_sparse_moe.experts.*.w1", "block_sparse_moe.experts.w1"],
                       ["block_sparse_moe.experts.*.w2", "block_sparse_moe.experts.w2"],
                       ["block_sparse_moe.gate"]]
layer_keys_llama_mlp_swiglu = [["mlp.swiglu.w12"],
                               ["mlp.swiglu.w3"]]
layer_keys_starcoder2_mlp = [["mlp.c_fc"],
                             ["mlp.c_proj"]]
layer_keys_gpt2_mlp = [["mlp.c_fc"],
                       ["mlp.c_proj"]]
expect_keys_llama = [["lm_head"],
                     ["model.norm"],
                     ["model.embed_tokens"]]
expect_keys_gemma = [["model.norm"],
                     ["model.embed_tokens"]]
expect_keys_starcoder2 = [["model.norm"],
                          ["model.embed_tokens"]]
expect_keys_gpt2 = [["model.embed_tokens"]]

dbrx_keymap = [("transformer.", "model."),
               (".blocks.", ".layers."),
               (".ffn.experts.mlp.", ".block_sparse_moe.experts."),
               (".ffn.router.layer.", ".block_sparse_moe.gate."),
               (".norm_attn_norm.norm_1.", ".input_layernorm."),
               (".norm_attn_norm.norm_2.", ".post_attention_layernorm."),
               (".norm_attn_norm.attn.", ".self_attn."),
               (".out_proj.", ".o_proj."),
               (".norm_f.", ".norm."),
               (".wte.", ".embed_tokens.")]
bigcode_keymap = [("transformer.ln_f", "model.norm"),
                  ("transformer.", "model."),
                  (".attn.c_proj.", ".self_attn.o_proj."),
                  (".attn.", ".self_attn."),
                  (".h.", ".layers."),
                  (".wte.", ".embed_tokens.")]
gpt2_keymap = [("$ln_f.", "model.norm."),
               (".attn.c_proj.", ".self_attn.o_proj."),
               (".attn.", ".self_attn."),
               ("$h.", "model.layers."),
               ("$wte.", "model.embed_tokens."),
               ("$wpe.", "model.wpe.")]
internlm2_keymap = [("$output.", "lm_head."),
                    ("$model.tok_embeddings.", "model.embed_tokens."),
                    (".attention.", ".self_attn."),
                    (".wo.", ".o_proj.")]

class RopeStyle(IntEnum):
    NONE = 0
    GPTJ = 1
    NEOX = 2

class ExLlamaV2ArchParams:

    def __init__(self, arch_string, read_config):

        self.arch_string = arch_string
        arch_recognized = False

        # Keys to expect in model dict
        self.expect_keys = []

        # Keys to expect in model dict, per layer
        self.layer_keys = []

        # Map tensors in HF model to standard keys
        self.keymap = None

        # Fused tensors
        self.fused_qkv_key = None
        self.fused_mlp_key_12 = None
        self.fused_mlp_key_3 = None

        # Alternate packing scheme for fused QKV tensor (InternLM2 quirk)
        self.fused_qkv_altpack = False

        # Learned position embeddings
        self.learned_pos_emb_key = None

        # Default multiplier for MLP inner dim (GPT2 quirk)
        self.default_inner_dim_mult = None

        # Compute logit scale from `dim_model_base` key in config.json (MiniCPM quirk)
        self.logit_scale_basedim = False

        # Tensors are transposed in original model weights
        self.orig_weights_transposed = False

        # Post norm keys
        self.norm_key_1_post = None
        self.norm_key_2_post = None

        # SWA required by architecture
        self.swa = False
        self.alternating_swa = False

        # Model only works with eager attention
        self.eager_attn_only = False

        # Clamp hidden states to FP16 range
        self.clamp_hidden_states = False

        # Upcast hidden state to FP32 before adding to residual stream
        self.residual_stream_fp32 = False

        # Expect bias for linear layers
        self.attention_bias_qkv = False
        self.attention_bias_o = False
        self.mlp_bias = False

        # Use gated MLP
        self.mlp_gate = True

        # Use block-sparse MLP
        self.is_moe = False

        # Normalize embeddings (Gemma quirk)
        self.normalize_embeddings = False

        # Constant bias for layernorm (Gemma quirk)
        self.norm_constant_bias = 0

        # Use parallel decoder blocks (Cohere quirk)
        self.parallel_decoder_blocks = False

        # Model is incoherent without BOS at the start of the context
        self.requires_bos = False

        # Use MQA, effectively num_key_valu_heads = 1 (GPTBigCode quirk)
        self.mqa = False

        # Scale attn weights (GPT2 quirk, not important for inference)
        self.scale_attn_weights = False

        # Model implementation works in tensor-parallel mode
        self.supports_tp = False

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
            self.mlp_key_gate = ".mlp.gate_proj"
            self.mlp_key_up = ".mlp.up_proj"
            self.mlp_key_down = ".mlp.down_proj"
            self.lm_head_key = "lm_head"
            self.norm_key_1 = ".input_layernorm"
            self.norm_key_2 = ".post_attention_layernorm"
            self.mlp_act_func = "silu"
            self.norm = "rmsnorm"
            self.rope_style = RopeStyle.NEOX
            self.supports_tp = True

        # Mixtral

        if arch_string == "MixtralForCausalLM":
            arch_recognized = True
            self.layer_keys += \
                layer_keys_llama_norms + \
                layer_keys_llama_attn + \
                layer_keys_mixtral_mlp
            self.expect_keys += \
                expect_keys_llama
            self.norm_eps_key = "rms_norm_eps"
            self.mlp_key_gate = ".block_sparse_moe.experts.*.w1"
            self.mlp_key_up = ".block_sparse_moe.experts.*.w3"
            self.mlp_key_down = ".block_sparse_moe.experts.*.w2"
            self.mlp_key_expert_gate = ".block_sparse_moe.gate"
            self.lm_head_key = "lm_head"
            self.norm_key_1 = ".input_layernorm"
            self.norm_key_2 = ".post_attention_layernorm"
            self.mlp_act_func = "silu"
            self.norm = "rmsnorm"
            self.rope_style = RopeStyle.NEOX
            self.is_moe = True

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
            self.mlp_key_gate = ".mlp.gate_proj"
            self.mlp_key_up = ".mlp.up_proj"
            self.mlp_key_down = ".mlp.down_proj"
            self.mlp_act_func = "silu"
            self.norm_key_1 = ".ln1"
            self.norm_key_2 = ".ln2"
            self.norm = "rmsnorm"
            self.lm_head_key = "lm_head"
            self.rope_style = RopeStyle.NEOX

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
            self.mlp_key_gate = ".mlp.gate_proj"
            self.mlp_key_up = ".mlp.up_proj"
            self.mlp_key_down = ".mlp.down_proj"
            self.lm_head_key = "lm_head"
            self.norm_key_1 = ".input_layernorm"
            self.norm_key_2 = ".post_attention_layernorm"
            self.mlp_act_func = "silu"
            self.norm = "layernorm"
            self.rope_style = RopeStyle.NEOX

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
            self.mlp_key_gate = ".mlp.gate_proj"
            self.mlp_key_up = ".mlp.up_proj"
            self.mlp_key_down = ".mlp.down_proj"
            self.lm_head_key = "lm_head"
            self.norm_key_1 = ".input_layernorm"
            self.norm_key_2 = ".post_attention_layernorm"
            self.mlp_act_func = "silu"
            self.norm = "rmsnorm"
            self.rope_style = RopeStyle.NEOX
            self.attention_bias_qkv = True
            self.supports_tp = True

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
            self.mlp_key_gate = ".mlp.gate_proj"
            self.mlp_key_up = ".mlp.up_proj"
            self.mlp_key_down = ".mlp.down_proj"
            self.lm_head_key = "model.embed_tokens"
            self.norm_key_1 = ".input_layernorm"
            self.norm_key_2 = ".post_attention_layernorm"
            self.mlp_act_func = "gelu"
            self.norm = "rmsnorm"
            self.rope_style = RopeStyle.NEOX
            self.normalize_embeddings = True
            self.norm_constant_bias = 1
            self.requires_bos = True

        # Gemma2

        if arch_string == "Gemma2ForCausalLM":
            arch_recognized = True
            self.layer_keys += \
                layer_keys_gemma2_norms + \
                layer_keys_llama_attn + \
                layer_keys_llama_mlp
            self.expect_keys += \
                expect_keys_gemma
            self.norm_eps_key = "rms_norm_eps"
            self.mlp_key_gate = ".mlp.gate_proj"
            self.mlp_key_up = ".mlp.up_proj"
            self.mlp_key_down = ".mlp.down_proj"
            self.lm_head_key = "model.embed_tokens"
            self.norm_key_1 = ".input_layernorm"
            self.norm_key_1_post = ".post_attention_layernorm"
            self.norm_key_2 = ".pre_feedforward_layernorm"
            self.norm_key_2_post = ".post_feedforward_layernorm"
            self.mlp_act_func = "gelu"
            self.norm = "rmsnorm"
            self.rope_style = RopeStyle.NEOX
            self.normalize_embeddings = True
            self.norm_constant_bias = 1
            self.requires_bos = True
            self.pre_post_layernorm = True
            self.alternating_swa = True
            self.residual_stream_fp32 = True

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
            self.mlp_key_up = ".mlp.c_fc"
            self.mlp_key_down = ".mlp.c_proj"
            self.lm_head_key = "model.embed_tokens"
            self.norm_key_1 = ".input_layernorm"
            self.norm_key_2 = ".post_attention_layernorm"
            self.mlp_act_func = "gelu"
            self.norm = "layernorm"
            self.rope_style = RopeStyle.NEOX
            self.attention_bias_qkv = True
            self.attention_bias_o = True
            self.mlp_bias = True
            self.mlp_gate = False

        # GemMoE

        if arch_string == "GemmoeForCausalLM":
            arch_recognized = True
            print(f" !! Warning, Gemmoe support is experimental and has not been fully tested")
            self.layer_keys += \
                layer_keys_llama_norms + \
                layer_keys_llama_attn + \
                layer_keys_mixtral_mlp
            self.expect_keys += \
                expect_keys_gemma
            self.norm_eps_key = "rms_norm_eps"
            self.mlp_key_gate = ".block_sparse_moe.experts.*.w1"
            self.mlp_key_up = ".block_sparse_moe.experts.*.w3"
            self.mlp_key_down = ".block_sparse_moe.experts.*.w2"
            self.mlp_key_expert_gate = ".block_sparse_moe.gate"
            self.lm_head_key = "model.embed_tokens"
            self.norm_key_1 = ".input_layernorm"
            self.norm_key_2 = ".post_attention_layernorm"
            self.mlp_act_func = "gelu"
            self.norm = "rmsnorm"
            self.rope_style = RopeStyle.NEOX
            self.normalize_embeddings = True
            self.norm_constant_bias = 1
            self.is_moe = True
            self.requires_bos = True

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
            self.mlp_key_gate = ".mlp.gate_proj"
            self.mlp_key_up = ".mlp.up_proj"
            self.mlp_key_down = ".mlp.down_proj"
            self.lm_head_key = "model.embed_tokens"
            self.norm_key_1 = ".input_layernorm"
            self.norm_key_2 = None
            self.mlp_act_func = "silu"
            self.norm = "layernorm"
            self.rope_style = RopeStyle.GPTJ
            self.parallel_decoder_blocks = True
            self.requires_bos = True

        # DBRX

        if arch_string == "DbrxForCausalLM":
            arch_recognized = True
            self.keymap = dbrx_keymap
            self.layer_keys += \
                layer_keys_llama_norms + \
                layer_keys_dbrx_attn + \
                layer_keys_dbrx_mlp
            self.expect_keys += \
                expect_keys_llama
            self.norm_eps_key = None
            self.mlp_key_gate = ".block_sparse_moe.experts.*.w1"
            self.mlp_key_up = ".block_sparse_moe.experts.*.v1"
            self.mlp_key_down = ".block_sparse_moe.experts.*.w2"
            self.mlp_key_expert_gate = ".block_sparse_moe.gate"
            self.lm_head_key = "lm_head"
            self.norm_key_1 = ".input_layernorm"
            self.norm_key_2 = ".post_attention_layernorm"
            self.mlp_act_func = "silu"
            self.norm = "layernorm"
            self.rope_style = RopeStyle.NEOX
            self.fused_qkv_key = "Wqkv"
            self.is_moe = True

        # Phi3

        if arch_string == "Phi3ForCausalLM":
            arch_recognized = True
            self.layer_keys += \
                layer_keys_llama_norms + \
                layer_keys_phi3_attn + \
                layer_keys_phi3_mlp
            self.expect_keys += \
                expect_keys_llama
            self.norm_eps_key = "rms_norm_eps"
            self.mlp_key_gate = ".mlp.gate_proj"
            self.mlp_key_up = ".mlp.up_proj"
            self.mlp_key_down = ".mlp.down_proj"
            self.lm_head_key = "lm_head"
            self.norm_key_1 = ".input_layernorm"
            self.norm_key_2 = ".post_attention_layernorm"
            self.fused_qkv_key = "qkv_proj"
            self.fused_mlp_key_12 = "gate_up_proj"
            self.mlp_act_func = "silu"
            self.norm = "rmsnorm"
            self.rope_style = RopeStyle.NEOX

        # GPTBigCode

        if arch_string == "GPTBigCodeForCausalLM":
            arch_recognized = True
            self.keymap = bigcode_keymap
            self.layer_keys += \
                layer_keys_gpt2_norms + \
                layer_keys_gpt2_attn + \
                layer_keys_gpt2_mlp
            self.expect_keys += \
                expect_keys_gpt2
            self.norm_eps_key = "layer_norm_epsilon"
            self.mlp_key_gate = None
            self.mlp_key_up = ".mlp.c_fc"
            self.mlp_key_down = ".mlp.c_proj"
            self.lm_head_key = "model.embed_tokens"
            self.norm_key_1 = ".ln_1"
            self.norm_key_2 = ".ln_2"
            self.fused_qkv_key = "c_attn"
            self.learned_pos_emb_key = "model.wpe"
            self.mlp_act_func = "gelu"
            self.norm = "layernorm"
            self.rope_style = RopeStyle.NONE
            self.mqa = True
            self.attention_bias_qkv = True
            self.attention_bias_o = True
            self.mlp_bias = True
            self.mlp_gate = False

        # GPT2

        if arch_string == "GPT2LMHeadModel":
            arch_recognized = True
            self.keymap = gpt2_keymap
            self.layer_keys += \
                layer_keys_gpt2_norms + \
                layer_keys_gpt2_attn + \
                layer_keys_gpt2_mlp
            self.expect_keys += \
                expect_keys_gpt2
            self.norm_eps_key = "layer_norm_epsilon"
            self.mlp_key_gate = None
            self.mlp_key_up = ".mlp.c_fc"
            self.mlp_key_down = ".mlp.c_proj"
            self.lm_head_key = "model.embed_tokens"
            self.norm_key_1 = ".ln_1"
            self.norm_key_2 = ".ln_2"
            self.fused_qkv_key = "c_attn"
            self.learned_pos_emb_key = "model.wpe"
            self.mlp_act_func = "gelu"
            self.norm = "layernorm"
            self.rope_style = RopeStyle.NONE
            self.default_inner_dim_mult = 4
            self.orig_weights_transposed = True
            self.attention_bias_qkv = True
            self.attention_bias_o = True
            self.mlp_bias = True
            self.mlp_gate = False

        # MiniCPM

        if arch_string == "MiniCPMForCausalLM":
            arch_recognized = True
            self.layer_keys += \
                layer_keys_llama_norms + \
                layer_keys_llama_attn + \
                layer_keys_llama_mlp
            self.expect_keys += \
                expect_keys_llama
            self.norm_eps_key = "rms_norm_eps"
            self.mlp_key_gate = ".mlp.gate_proj"
            self.mlp_key_up = ".mlp.up_proj"
            self.mlp_key_down = ".mlp.down_proj"
            self.lm_head_key = "lm_head"
            self.norm_key_1 = ".input_layernorm"
            self.norm_key_2 = ".post_attention_layernorm"
            self.mlp_act_func = "silu"
            self.norm = "rmsnorm"
            self.rope_style = RopeStyle.NEOX
            self.logit_scale_basedim = True

        # InternLM2

        if arch_string == "InternLM2ForCausalLM":
            arch_recognized = True
            self.keymap = internlm2_keymap
            self.layer_keys += \
                layer_keys_internlm2_norms + \
                layer_keys_internlm2_attn + \
                layer_keys_internlm2_mlp
            self.expect_keys += \
                expect_keys_llama
            self.norm_eps_key = "rms_norm_eps"
            self.mlp_key_gate = ".feed_forward.w1"
            self.mlp_key_up = ".feed_forward.w3"
            self.mlp_key_down = ".feed_forward.w2"
            self.lm_head_key = "lm_head"
            self.norm_key_1 = ".attention_norm"
            self.norm_key_2 = ".ffn_norm"
            self.fused_qkv_key = "wqkv"
            self.mlp_act_func = "silu"
            self.norm = "rmsnorm"
            self.rope_style = RopeStyle.NEOX
            self.fused_qkv_altpack = True

        # Index

        if arch_string == "IndexForCausalLM":
            arch_recognized = True
            self.layer_keys += \
                layer_keys_llama_norms + \
                layer_keys_llama_attn + \
                layer_keys_llama_mlp
            self.expect_keys += \
                expect_keys_llama
            self.norm_eps_key = "rms_norm_eps"
            self.mlp_key_gate = ".mlp.gate_proj"
            self.mlp_key_up = ".mlp.up_proj"
            self.mlp_key_down = ".mlp.down_proj"
            self.lm_head_key = "lm_head"
            self.norm_key_1 = ".input_layernorm"
            self.norm_key_2 = ".post_attention_layernorm"
            self.mlp_act_func = "silu"
            self.norm = "rmsnorm"
            self.rope_style = RopeStyle.NEOX

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
            self.mlp_key_gate = ".mlp.gate_proj"
            self.mlp_key_up = ".mlp.up_proj"
            self.mlp_key_down = ".mlp.down_proj"
            self.lm_head_key = "lm_head"
            self.norm_key_1 = ".input_layernorm"
            self.norm_key_2 = ".post_attention_layernorm"
            self.mlp_act_func = "silu"
            self.norm = "rmsnorm"
            self.rope_style = RopeStyle.NEOX
            self.supports_tp = True

        # Arch overrides

        if read_config.get("attention_bias", False):
            self.attention_bias_qkv = True
            self.attention_bias_o = True

        if read_config.get("mlp_bias", False):
            self.mlp_bias = True

        if read_config.get("tie_word_embeddings", False):
            if ["lm_head"] in self.expect_keys:
                self.expect_keys.remove(["lm_head"])
                self.lm_head_key = "model.embed_tokens"

        # Sanity checks

        if self.residual_stream_fp32:
            assert self.norm_key_1_post and self.norm_key_2_post, \
                "FP32 residual stream only implemented for arch with post layernorms"

    def make_fused_mlp(self):

        for x in layer_keys_llama_mlp: self.layer_keys.remove(x)
        self.layer_keys += layer_keys_llama_mlp_swiglu
        self.fused_mlp_key_12 = layer_keys_llama_mlp_swiglu[0][0]
        self.fused_mlp_key_3 = layer_keys_llama_mlp_swiglu[1][0]


