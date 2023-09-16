from exllamav2.model import ExLlamaV2Embedding, ExLlamaV2Attention, ExLlamaV2MLP, ExLlamaV2Linear
from safetensors import safe_open
from safetensors.torch import save_file
from conversion.qparams import QParams, qparams_options, qparams_headoptions
from conversion.adaptivegptq import AdaptiveGPTQ
import torch
from torch import nn
import os, time, math, json
import torch.nn.functional as F
import gc

def list_live_tensors():

    tensors = {}
    gc.collect()
    torch.cuda.empty_cache()

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                d = str(obj.size()) + ", " + str(obj.dtype) + ", " + str(obj.device)
                if d in tensors.keys():
                    tensors[d] += 1
                else:
                    tensors[d] = 1
        except:
            pass

    print("-----------")
    for k, v in tensors.items():
        print(f"{v} : {k}")


# Get initial token embeddings

def embeddings(job, save_fn, model, measure = False):

    module = model.modules[0]
    assert isinstance(module, ExLlamaV2Embedding)

    with safe_open(job["cal_filename"], framework = "pt", device = "cpu") as f:
        input_ids = f.get_tensor("input_ids")

    module.load()
    hidden_state = module.forward(input_ids)
    module.unload()

    embeddings_dict = { "hidden_state": hidden_state }
    save_file(embeddings_dict, os.path.join(job["out_dir"], "input_states.safetensors"))


# Measure quantization error as relative Frobenius norm wrt/ inputs and full-precision outputs

def rfn_error(q_linear: nn.Linear, inputs, outputs):

    dsum = 0.0
    dcount = 0.0
    for ix, ox in zip(inputs, outputs):

        ix_cuda = ix.to("cuda:0")
        ox_cuda = ox.to("cuda:0")

        qx_cuda = q_linear.forward(ix_cuda)
        rfn = torch.linalg.norm(qx_cuda[0].float() - ox_cuda[0].float(), 'fro') / torch.linalg.norm(ox_cuda[0].float(), 'fro')

        dsum += rfn * ix_cuda.shape[0]
        dcount += ix_cuda.shape[0]

        ix_cuda = None
        ox_cuda = None

    return dsum / dcount


# Measure quantization impact per layer

def test_quants(source: ExLlamaV2Linear,
                lq: AdaptiveGPTQ,
                inputs: list,
                outputs: list,
                qparams: list,
                results: list,
                skip_prep: bool = False):

    with torch.inference_mode():

        time_a = time.time()

        print(f" -- Linear: {source.key}")
        result = { "key": source.key,
                   "numel": source.in_features * source.out_features,
                   "options": [] }

        original = nn.Linear(source.in_features, source.out_features, False, device = "meta", dtype = torch.float16)
        original.weight = nn.Parameter(source.linear.weight.clone())

        # lq = AdaptiveGPTQ(original)

        # b = 0
        # while b < len(inputs):
        #     a = b
        #     b = min(b + 8, len(inputs))
        #     inputs_cuda = inputs[a:b]
        #     lq.add_batch(inputs_cuda)
        #     inputs_cuda = None

        if not skip_prep: lq.prepare()

        for qp in qparams:

            lq.configure(qp.group_size, qp.bits, qp.bits_prop, qp.scale_bits)
            lq.quantize()

            quantized = lq.apply_temp()
            bpw = qp.bpw(quantized.weight.T.shape)
            desc = qp.desc
            err = rfn_error(quantized, inputs, outputs).item()

            print(f" -- {desc:30} {bpw:2.2f} bpw    rfn_error: {err:2.5f}")

            option = { "desc": desc,
                       "bpw": bpw,
                       "total_bits": lq.rows * lq.columns * bpw,
                       "err": err,
                       "qparams": qp.get_dict() }
            result["options"].append(option)

        results.append(result)

        time_b = time.time()
        print(f" -- Time: {time_b - time_a:.2f} seconds")



def measure_quant(job, save_fn, model):

    # Quantize

    if not "last_module_idx" in job:
        job["last_module_idx"] = 0

    input_states = None
    output_states = None

    page_rows = (job["gpu_rows"] < job["measurement_rows"])

    while True:

        index = job["last_module_idx"]
        index += 1
        if index >= len(model.modules): break

        # Prepare module

        module = model.modules[index]
        module.load()

        print(f" -- Layer: {module.key} ({module.name})")

        # Reference forward pass

        in_name = os.path.join(job["out_dir"], "input_states.safetensors")
        out_name = os.path.join(job["out_dir"], "output_states.safetensors")

        if output_states is not None:
            input_states = output_states
            output_states = None
        else:
            with safe_open(in_name, framework = "pt", device = "cpu" if page_rows else "cuda:0") as f:
                input_states = f.get_tensor("hidden_state")

        with torch.inference_mode():

            output_states_list = []
            all_outputs_list = []
            quantizers = {}
            results = None

            batchsize = 1
            batch1 = []
            batch2 = []

            for b in range(input_states.shape[0]):

                last = (b == input_states.shape[0] - 1)

                x = input_states[b:b+1, :, :].to("cuda:0")
                cache = None
                attn_mask = None
                if isinstance(module, ExLlamaV2Attention):
                    attn_mask = model.build_attn_mask(1, x.shape[1], 0, None, "cuda:0")

                outputs = module.forward(x, cache, attn_mask, intermediates = True)

                for k, v in outputs.items():
                    v[v == -float('inf')] = -65504.0
                    v[v == float('inf')] = 65504.0

                if page_rows:
                    for k in outputs.keys(): outputs[k] = outputs[k].to("cpu")

                if isinstance(module, ExLlamaV2Attention):
                    if not "q_proj" in quantizers: quantizers["q_proj"] = AdaptiveGPTQ(module.q_proj.linear)
                    if not "k_proj" in quantizers: quantizers["k_proj"] = AdaptiveGPTQ(module.k_proj.linear)
                    if not "v_proj" in quantizers: quantizers["v_proj"] = AdaptiveGPTQ(module.v_proj.linear)
                    if not "o_proj" in quantizers: quantizers["o_proj"] = AdaptiveGPTQ(module.o_proj.linear)

                    batch1.append(outputs["post_norm"])
                    if len(batch1) == batchsize or last:
                        quantizers["q_proj"].add_batch(batch1)
                        batch1 = []

                    batch2.append(outputs["attn_output"])
                    if len(batch2) == batchsize or last:
                        quantizers["o_proj"].add_batch(batch2)
                        batch2 = []

                elif isinstance(module, ExLlamaV2MLP):
                    if not "gate_proj" in quantizers: quantizers["gate_proj"] = AdaptiveGPTQ(module.gate_proj.linear)
                    if not "up_proj"   in quantizers: quantizers["up_proj"  ] = AdaptiveGPTQ(module.up_proj.linear)
                    if not "down_proj" in quantizers: quantizers["down_proj"] = AdaptiveGPTQ(module.down_proj.linear)

                    batch1.append(outputs["post_norm"])
                    if len(batch1) == batchsize or last:
                        quantizers["gate_proj"].add_batch(batch1)
                        batch1 = []

                    batch2.append(outputs["pre_down"])
                    if len(batch2) == batchsize or last:
                        quantizers["down_proj"].add_batch(batch2)
                        batch2 = []

                # elif module.key == "lm_head":
                #     if not "lm_head" in quantizers: quantizers["lm_head"] = AdaptiveGPTQ(module.linear)
                #     quantizers["lm_head"].add_batch([x])

                output_states_list.append(outputs["hidden_states"])
                del outputs["hidden_states"]
                all_outputs_list.append(outputs)
                outputs = None
                attn_mask = None
                x = None

            output_states = torch.cat(output_states_list, dim = 0)
            output_states_list = None
            input_states = None
            save_file({ "hidden_state": output_states }, out_name)

            # Attention layer

            if isinstance(module, ExLlamaV2Attention):

                results = []

                post_norm     = [x["post_norm"] for x in all_outputs_list]
                query_states  = [x["query_states"] for x in all_outputs_list]
                key_states    = [x["key_states"] for x in all_outputs_list]
                value_states  = [x["value_states"] for x in all_outputs_list]
                attn_output   = [x["attn_output"] for x in all_outputs_list]
                attn_proj     = [x["attn_proj"] for x in all_outputs_list]

                all_outputs_list = None
                torch.cuda.empty_cache()

                test_quants(module.q_proj, quantizers["q_proj"], post_norm, query_states, qparams_options, results)
                quantizers["k_proj"].reuse_h(quantizers["q_proj"])
                quantizers["v_proj"].reuse_h(quantizers["q_proj"])
                del quantizers["q_proj"]
                torch.cuda.empty_cache()

                test_quants(module.k_proj, quantizers["k_proj"], post_norm, key_states, qparams_options, results, skip_prep = True)
                del quantizers["k_proj"]
                torch.cuda.empty_cache()

                test_quants(module.v_proj, quantizers["v_proj"], post_norm, value_states, qparams_options, results, skip_prep = True)
                post_norm = None
                del quantizers["v_proj"]
                torch.cuda.empty_cache()

                test_quants(module.o_proj, quantizers["o_proj"], attn_output, attn_proj, qparams_options, results)
                del quantizers["o_proj"]
                query_states = None
                key_states = None
                value_states = None
                attn_output = None
                attn_proj = None
                torch.cuda.empty_cache()

            # MLP layer

            if isinstance(module, ExLlamaV2MLP):

                results = []

                post_norm     = [x["post_norm"] for x in all_outputs_list]
                gate          = [x["gate"] for x in all_outputs_list]
                up            = [x["up"] for x in all_outputs_list]
                pre_down      = [x["pre_down"] for x in all_outputs_list]
                down          = [x["down"] for x in all_outputs_list]

                all_outputs_list = None

                test_quants(module.gate_proj, quantizers["gate_proj"], post_norm, gate, qparams_options, results)
                quantizers["up_proj"].reuse_h(quantizers["gate_proj"])
                del quantizers["gate_proj"]
                gate = None
                torch.cuda.empty_cache()

                test_quants(module.up_proj, quantizers["up_proj"], post_norm, up, qparams_options, results, skip_prep = True)
                del quantizers["up_proj"]
                up = None
                post_norm = None
                torch.cuda.empty_cache()

                test_quants(module.down_proj, quantizers["down_proj"], pre_down, down, qparams_options, results)
                del quantizers["down_proj"]
                pre_down = None
                down = None
                torch.cuda.empty_cache()

            # Free up some VRAM

            all_outputs_list = None
            torch.cuda.empty_cache()

            # Head module

            if module.key == "lm_head":

                if module.padding > 0: output_states = output_states[:, :, :-module.padding]

                with safe_open(job["cal_filename"], framework = "pt", device = "cpu") as f:
                    cal_ids = f.get_tensor("input_ids")

                with torch.inference_mode():

                    logprob_sum = 0.0
                    logprob_count = 0

                    for i in range(output_states.shape[0]):

                        logits = output_states[i:i+1, :, :].to("cuda:0")
                        ids = cal_ids[i]

                        target_ids = ids.unsqueeze(0)[:, 1:].to("cuda:0")
                        logits = logits[:, :-1, :]

                        log_probs = F.log_softmax(logits, dim = -1)
                        token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
                        logprob_sum += token_log_probs.sum().item()
                        logprob_count += target_ids.numel()

                    mean_log_prob = logprob_sum / logprob_count
                    perplexity = math.exp(-mean_log_prob)

                    print(f" -- Calibration perplexity (base): {perplexity:.4f}")
                    job["base_perplexity"] = perplexity

            # Unload module

            module.unload()

        # Advance

        job["invalid"] = True
        save_fn()

        if results is not None:
            if not "measurement" in job:
                job["measurement"] = []
            job["measurement"] += results

        job["last_module_idx"] = index

        os.remove(in_name)
        os.rename(out_name, in_name)

        del job["invalid"]
        save_fn()

    # Export measurement

    exp_measurement = { "measurement": job["measurement"],
                        "last_module_idx": job["last_module_idx"],
                        "base_perplexity": job["base_perplexity"] }

    with open(os.path.join(job["out_dir"], "measurement.json"), "w") as f:
        f.write(json.dumps(exp_measurement, indent = 4))


# Quantize

def do_quant(source: ExLlamaV2Linear,
             lq: AdaptiveGPTQ,
             qparams: dict,
             job: dict,
             skip_prep: bool = False):

    with torch.inference_mode():

        qp = QParams.from_dict(qparams)
        print(f" -- Linear: {source.key} -> {qp.get_desc()}, {qp.bpw(source.linear.weight.T.shape):.2f} bpw")

        # Prepare quantizer

        if not skip_prep: lq.prepare()
        lq.configure(qp.group_size, qp.bits, qp.bits_prop, qp.scale_bits)

        # Perform final quant

        lq.quantize(keep_qweight = True)

        # Sanity test to ensure quantized matrix resembles original

        # mat_error_1, mat_error_5, mat_error_10 = lq.quant_error()
        # print(f" -- %1+: {mat_error_1:.6f}  %5+: {mat_error_5:.6f}  %10+: {mat_error_10:.6f} ")
        # if mat_error_5 > 0.01:
        #
        #     print(" ## Quantization error (1)")
        #     os._exit(0)

        # Apply quant

        lq.apply_quant()

        # Pack and save quantized layer

        packed_dict = lq.pack(source.key, qp)
        tensorfile = os.path.join(job["out_dir"], "out_tensor/" + source.key + ".safetensors")
        save_file(packed_dict, tensorfile)

        # Reconstruct from packed layer

        recons_linear = ExLlamaV2Linear(source.model, source.key, source.in_features, source.out_features, False)
        recons_linear.device_idx = source.device_idx
        recons_dict = {}
        for k in ["q_weight", "q_invperm", "q_scale", "q_scale_max", "q_groups"]:
            recons_dict[k] = packed_dict[source.key + "." + k]
        recons_dict["q_perm"] = torch.argsort(recons_dict["q_invperm"]).to(torch.int)
        recons_linear.load(recons_dict)

        # Sanity test to ensure reconstructed matrix matches unpacked matrix

        quant_w = source.linear.weight.T
        recons_w = recons_linear.get_weight_tensor_dq()

        ident = torch.eye(recons_linear.in_features, dtype = torch.half).cuda()
        recons_w2 = recons_linear.forward(ident, force_cuda = True)

        diff1 = torch.max(torch.abs(quant_w - recons_w))
        diff2 = torch.max(torch.abs(quant_w - recons_w2))
        if diff1 > 0.01 or diff2 > 0.01:

            print(" ## Quantization error (2)")
            os._exit(0)

        # Apply reconstructed matrix to source layer

        source.linear.weight.data = recons_w.T


def quant(job, save_fn, model):

    qparams = {}
    for layer in job["measurement"]:
        qparams[layer["key"]] = layer["best_option"]["qparams"]

    # Quantize

    if not "q_last_module_idx" in job:
        job["q_last_module_idx"] = 0

    input_states = None
    output_states = None

    page_rows = (job["gpu_rows"] < job["dataset_rows"])

    while True:

        index = job["q_last_module_idx"]
        index += 1
        if index >= len(model.modules): break

        # Prepare module

        module = model.modules[index]
        print(f" -- Layer: {module.key} ({module.name})")
        module.load()

        time_begin = time.time()

         # Reference forward pass

        in_name = os.path.join(job["out_dir"], "input_states.safetensors")

        if output_states is not None:
            input_states = output_states
            output_states = None
        else:
            with safe_open(in_name, framework = "pt", device = "cpu" if page_rows else "cuda:0") as f:
                input_states = f.get_tensor("hidden_state")

        output_states_list = []
        quantizers = {}

        with torch.inference_mode():

            batchsize = 8
            batch1 = []
            batch2 = []

            for b in range(input_states.shape[0]):

                last = (b == input_states.shape[0] - 1)

                x = input_states[b:b+1, :, :].to("cuda:0")
                cache = None
                attn_mask = None
                if isinstance(module, ExLlamaV2Attention):
                    attn_mask = model.build_attn_mask(1, x.shape[1], 0, None, "cuda:0")

                outputs = module.forward(x, cache, attn_mask, intermediates = True)

                # Clamp state values to FP16 range

                for k, v in outputs.items():
                    v[v == -float('inf')] = -65504.0
                    v[v == float('inf')] = 65504.0

                # Add batches to quantizers

                if isinstance(module, ExLlamaV2Attention):
                    if not "q_proj" in quantizers: quantizers["q_proj"] = AdaptiveGPTQ(module.q_proj.linear)
                    if not "k_proj" in quantizers: quantizers["k_proj"] = AdaptiveGPTQ(module.k_proj.linear)
                    if not "v_proj" in quantizers: quantizers["v_proj"] = AdaptiveGPTQ(module.v_proj.linear)
                    if not "o_proj" in quantizers: quantizers["o_proj"] = AdaptiveGPTQ(module.o_proj.linear)

                    batch1.append(outputs["post_norm"])
                    if len(batch1) == batchsize or last:
                        quantizers["q_proj"].add_batch(batch1)
                        batch1 = []

                    batch2.append(outputs["attn_output"])
                    if len(batch2) == batchsize or last:
                        quantizers["o_proj"].add_batch(batch2)
                        batch2 = []

                elif isinstance(module, ExLlamaV2MLP):
                    if not "gate_proj" in quantizers: quantizers["gate_proj"] = AdaptiveGPTQ(module.gate_proj.linear)
                    if not "up_proj"   in quantizers: quantizers["up_proj"  ] = AdaptiveGPTQ(module.up_proj.linear)
                    if not "down_proj" in quantizers: quantizers["down_proj"] = AdaptiveGPTQ(module.down_proj.linear)

                    batch1.append(outputs["post_norm"])
                    if len(batch1) == batchsize or last:
                        quantizers["gate_proj"].add_batch(batch1)
                        batch1 = []

                    batch2.append(outputs["pre_down"])
                    if len(batch2) == batchsize or last:
                        quantizers["down_proj"].add_batch(batch2)
                        batch2 = []

                elif module.key == "lm_head":
                    if not "lm_head" in quantizers: quantizers["lm_head"] = AdaptiveGPTQ(module.linear)

                    batch1.append(x)
                    if len(batch1) == batchsize or last:
                        quantizers["lm_head"].add_batch(batch1)
                        batch1 = []

                output_states_list.append(outputs["hidden_states"].to("cpu"))

                outputs = None
                attn_mask = None
                x = None

        # Attention layer

        if isinstance(module, ExLlamaV2Attention):

            do_quant(module.q_proj, quantizers["q_proj"], qparams[module.q_proj.key], job)
            quantizers["k_proj"].reuse_h(quantizers["q_proj"])
            quantizers["v_proj"].reuse_h(quantizers["q_proj"])
            del quantizers["q_proj"]
            torch.cuda.empty_cache()
            do_quant(module.k_proj, quantizers["k_proj"], qparams[module.k_proj.key], job, skip_prep = True)
            del quantizers["k_proj"]
            torch.cuda.empty_cache()
            do_quant(module.v_proj, quantizers["v_proj"], qparams[module.v_proj.key], job, skip_prep = True)
            del quantizers["v_proj"]
            torch.cuda.empty_cache()
            do_quant(module.o_proj, quantizers["o_proj"], qparams[module.o_proj.key], job)
            del quantizers["o_proj"]
            torch.cuda.empty_cache()

        # MLP layer

        if isinstance(module, ExLlamaV2MLP):

            do_quant(module.gate_proj, quantizers["gate_proj"], qparams[module.gate_proj.key], job)
            quantizers["up_proj"].reuse_h(quantizers["gate_proj"])
            del quantizers["gate_proj"]
            torch.cuda.empty_cache()
            do_quant(module.up_proj,   quantizers["up_proj"  ], qparams[module.up_proj.key  ], job, skip_prep = True)
            del quantizers["up_proj"]
            torch.cuda.empty_cache()
            do_quant(module.down_proj, quantizers["down_proj"], qparams[module.down_proj.key], job)
            del quantizers["down_proj"]
            torch.cuda.empty_cache()

        # Head module

        if module.key == "lm_head" and isinstance(module, ExLlamaV2Linear):

            bits = job["head_bits"]
            qp = qparams_headoptions[bits]
            if qp is not None:

                do_quant(module, quantizers["lm_head"], qp.get_dict(), job)
                del quantizers["lm_head"]
                torch.cuda.empty_cache()

            # Start computing perplexity on last layer

            with safe_open(job["cal_filename"], framework = "pt", device = "cpu") as f:
                cal_ids = f.get_tensor("input_ids")

            logprob_sum = 0.0
            logprob_count = 0

        # Post-quantization forward pass

        out_name = os.path.join(job["out_dir"], "output_states.safetensors")

        with torch.inference_mode():

            rfn_sum = 0.0

            for b in range(input_states.shape[0]):

                x = input_states[b:b+1, :, :].to("cuda:0")
                cache = None
                attn_mask = None
                if isinstance(module, ExLlamaV2Attention):
                    attn_mask = model.build_attn_mask(1, x.shape[1], 0, None, "cuda:0")

                outputs = module.forward(x, cache, attn_mask)

                # Clamp state values to FP16 range

                outputs[outputs == -float('inf')] = -65504.0
                outputs[outputs == float('inf')] = 65504.0

                # Compute perplexity for head layer without saving output state

                if module.key == "lm_head" and b < job["measurement_rows"]:

                    if module.padding > 0: outputs = outputs[:, :, :-module.padding]

                    logits = outputs[:, :-1, :]
                    target_ids = cal_ids[b].unsqueeze(0)[:, 1:].to("cuda:0")

                    log_probs = F.log_softmax(logits, dim = -1)
                    token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
                    logprob_sum += token_log_probs.sum().item()
                    logprob_count += target_ids.numel()

                else:

                    # Measure error

                    target = output_states_list[b]
                    if target.device == torch.device("cpu"): target = target.to("cuda:0")
                    a_ = outputs.narrow(-1, 0, target.shape[-1])
                    b_ = target
                    rfn = torch.linalg.norm(a_[0].float() - b_[0].float(), 'fro') / torch.linalg.norm(b_[0].float(), 'fro')

                    # print(f"b norm: {torch.linalg.norm(b_[0].float(), 'fro')}")
                    # print(f"a min, max: {torch.min(a_[0])}, {torch.max(a_[0])}")
                    # print(f"b min, max: {torch.min(b_[0])}, {torch.max(b_[0])}")

                    rfn_sum += rfn
                    target = None

                    if page_rows:
                        outputs = outputs.to("cpu")

                    output_states_list[b] = outputs

                outputs = None
                x = None
                attn_mask = None

            rfn_avg = rfn_sum / input_states.shape[0]
            print(f" -- Layer rfn_error: {rfn_avg:.6f}")

            if math.isnan(rfn_avg) or rfn_avg > 1.0:
                print(" ## Quantization error (3)")
                os._exit(0)

            if module.key != "lm_head":

                output_states = torch.cat(output_states_list, dim = 0)
                save_file({ "hidden_state": output_states }, out_name)

            input_states = None
            del input_states

        # Perplexity

        if module.key == "lm_head" and isinstance(module, ExLlamaV2Linear):

            mean_log_prob = logprob_sum / logprob_count
            perplexity = math.exp(-mean_log_prob)

            print(f" -- Calibration perplexity (quant): {perplexity:.4f}")
            job["cal_perplexity"] = perplexity

        # Unload module

        output_states_list = None
        module.unload()

        # Advance

        job["invalid"] = True
        save_fn()

        job["q_last_module_idx"] = index

        if module.key != "lm_head":
            os.remove(in_name)
            os.rename(out_name, in_name)

        if "invalid" in job: del job["invalid"]
        save_fn()

        # Report time taken

        time_end = time.time()
        layer_time = time_end - time_begin
        print(f" -- Module quantized, time: {layer_time:.2f} seconds")

    # Export measurement

    exp_measurement = { "measurement": job["measurement"],
                        "last_module_idx": job["last_module_idx"],
                        "base_perplexity": job["base_perplexity"] }

    with open(os.path.join(job["out_dir"], "measurement.json"), "w") as f:
        f.write(json.dumps(exp_measurement, indent = 4))

