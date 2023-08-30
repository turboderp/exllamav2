import os
from safetensors import safe_open
from safetensors.torch import save_file
from exllamav2.model import ExLlamaV2Attention, ExLlamaV2MLP, ExLlamaV2RMSNorm, ExLlamaV2Embedding, ExLlamaV2Linear


def get_f_module(job, module):

    mod_dict = {}
    module.load()
    mod_dict[module.key + ".weight"] = module.get_weight()
    return mod_dict


def get_q_module(job, module):

    mod_dict = {}
    filename = os.path.join(job["out_dir"], "out_tensor/" + module.key + ".safetensors")
    with safe_open(filename, framework = "pt", device = "cpu") as f:
        for k in f.keys():
            mod_dict[k] = f.get_tensor(k)
    return mod_dict


def compile_model(job, save_fn, model):

    out_dict = {}

    index = 0
    while index < len(model.modules):

        module = model.modules[index]

        if isinstance(module, ExLlamaV2Embedding):

            out_dict |= get_f_module(job, module)

        if isinstance(module, ExLlamaV2Attention):

            out_dict |= get_f_module(job, module.input_layernorm)
            out_dict |= get_q_module(job, module.q_proj)
            out_dict |= get_q_module(job, module.k_proj)
            out_dict |= get_q_module(job, module.v_proj)
            out_dict |= get_q_module(job, module.o_proj)

        if isinstance(module, ExLlamaV2MLP):

            out_dict |= get_f_module(job, module.post_attention_layernorm)
            out_dict |= get_q_module(job, module.gate_proj)
            out_dict |= get_q_module(job, module.up_proj)
            out_dict |= get_q_module(job, module.down_proj)

        if isinstance(module, ExLlamaV2RMSNorm):

            out_dict |= get_f_module(job, module)

        if isinstance(module, ExLlamaV2Linear):

            assert module.key == "lm_head"
            out_dict |= get_q_module(job, module)

        index += 1

    out_file = os.path.join(job["out_dir"], "output.safetensors")
    save_file(out_dict, out_file)