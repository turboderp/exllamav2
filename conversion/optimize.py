from conversion.qparams import QParams
import math
import itertools

def optimize(job, save_fn, model):

    has_gate = model.config.arch.mlp_gate
    if has_gate: mlp_key_gate = model.config.arch.mlp_key_gate
    mlp_key_up = model.config.arch.mlp_key_up
    mlp_key_down = model.config.arch.mlp_key_down

    error_norm = 2.4
    max_step_size = 2
    first_layer_bias = 10
    bias_layers = 2
    bias_iter = 10

    key = "model.layers.0"
    key_q = key + ".self_attn.q_proj"
    key_k = key + ".self_attn.k_proj"
    key_v = key + ".self_attn.v_proj"
    key_o = key + ".self_attn.o_proj"

    if not model.config.arch.is_moe:
        if has_gate: key_g = key + mlp_key_gate
        key_u = key + mlp_key_up
        key_d = key + mlp_key_down
        mlp_mode = "mlp"
    else:
        if has_gate: key_g = key + mlp_key_gate.replace("*", "0")
        key_u = key + mlp_key_up.replace("*", "0")
        key_d = key + mlp_key_down.replace("*", "0")
        mlp_mode = "block_sparse_moe"

    num_experts = model.config.num_experts if model.config.num_experts is not None else 1
    shape_q = model.modules_dict[key_q].matrix_shape()
    shape_k = model.modules_dict[key_k].matrix_shape()
    shape_v = model.modules_dict[key_v].matrix_shape()
    shape_o = model.modules_dict[key_o].matrix_shape()
    shape_g = model.modules_dict[key_g].matrix_shape() if has_gate else None
    shape_u = model.modules_dict[key_u].matrix_shape()
    shape_d = model.modules_dict[key_d].matrix_shape()
    numel_q = shape_q[0] * shape_q[1]
    numel_k = shape_k[0] * shape_k[1]
    numel_v = shape_v[0] * shape_v[1]
    numel_o = shape_o[0] * shape_o[1]
    numel_g = shape_g[0] * shape_g[1] * num_experts if has_gate else 0
    numel_u = shape_u[0] * shape_u[1] * num_experts
    numel_d = shape_d[0] * shape_d[1] * num_experts
    numel_attn = numel_q + numel_k + numel_v + numel_o
    numel_mlp = numel_g + numel_u + numel_d

    # Combined size of hidden layers

    num_layers = model.config.num_hidden_layers
    num_modules = num_layers * 2
    numel = sum(m.numel() for m in model.modules[1 : num_modules + 1])

    target_bpw = job["bits"]
    weight_budget = numel * target_bpw

    # Compile options

    measurement = job["measurement"]

    def fn(x, idx):
        if idx < bias_layers:
            return 1 - ((1 - x) ** error_norm) * first_layer_bias
        else:
            return 1 - ((1 - x) ** error_norm)

    weights = []
    values = []
    params = []
    for i in range(num_layers):
        if model.config.arch.parallel_decoder_blocks:
            m1 = measurement["model.layers." + str(i) + ".parallel_decoder"]["attn"]
            m2 = measurement["model.layers." + str(i) + ".parallel_decoder"]["mlp"]
        else:
            m1 = measurement["model.layers." + str(i) + ".self_attn"]
            m2 = measurement["model.layers." + str(i) + "." + mlp_mode]
        for m in [m1, m2]:
            v = [fn(e["accuracy"], i) for e in m]
            w = [e["total_bits"] for e in m]
            weights.append(w)
            values.append(v)
            params.append(m)

    print(" -- Pruning...")

    # Sort options by weight, eliminate strictly worse options

    for i in range(num_layers * 2):
        combined = sorted(zip(weights[i], values[i], params[i]))
        w_, v_, p_ = zip(*combined)
        w_ = list(w_)
        v_ = list(v_)
        p_ = list(p_)
        j = 1
        while j < len(v_):
            if v_[j] <= v_[j - 1]:
                w_.pop(j)
                v_.pop(j)
                p_.pop(j)
            else:
                j += 1
        weights[i] = w_
        values[i] = v_
        params[i] = p_

    # Quick and dirty iterative solver

    print(" -- Solving...")

    f_solution = [0] * num_layers * 2
    weight = sum(weights[i][0] for i in range(num_layers * 2))
    value = 1
    for i in range(num_layers * 2): value *= values[i][0]

    iteration = 0

    while True:
        min_idx = -1
        min_value = float("inf")
        iteration += 1
        for i in range(bias_layers if iteration < bias_iter else num_layers * 2):
            s = f_solution[i]
            if values[i][s] < min_value:
                if s < len(weights[i]) - 1:
                    added_w = weights[i][s + 1] - weights[i][s]
                    if added_w + weight <= weight_budget:
                        min_idx = i
                        min_value = values[i][s]
        if min_idx == -1: break
        s = f_solution[min_idx]
        weight += weights[min_idx][s + 1] - weights[min_idx][s]
        value *= values[min_idx][s + 1] / values[min_idx][s]
        f_solution[min_idx] += 1

    bpw = weight / numel
    print(f" -- Score: {value:.8f}  bpw: {bpw:.4f}")

    def improve(solution, s_weight, hold = None):

        if hold is None: hold = []
        best_idx = -1
        best_ratio = 0
        best_add_w = 0
        best_add_v = 0
        for idx in range(num_layers * 2):
            if idx in hold: continue

            si = solution[idx]
            if si == len(weights[idx]) - 1: continue

            add_w = weights[idx][si + 1] - weights[idx][si]
            if s_weight + add_w > weight_budget: continue

            add_v = values[idx][si + 1] / values[idx][si]
            ratio = add_v / add_w
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = idx
                best_add_w = add_w
                best_add_v = add_v

        return best_idx, best_add_w, best_add_v

    # while True:
    #     b_idx, b_add_w, b_add_v = improve(f_solution, weight)
    #     if b_idx == -1:
    #         break
    #
    #     f_solution[b_idx] += 1
    #     weight += b_add_w
    #     value += b_add_v
    #
    # bpw = weight / numel
    # print(f" -- Score: {math.exp(value):.8f}  bpw: {bpw:.4f}")

    best_value = value
    prev_best_value = value
    step_size = 1

    while True:

        for i, j in itertools.permutations(range(num_layers * 2), 2):

            t_solution = f_solution.copy()
            t_solution[i] = max(t_solution[i] - step_size, 0)
            t_solution[j] = max(t_solution[j] - step_size, 0)

            t_weight = sum(weights[k][t_solution[k]] for k in range(num_layers * 2))
            t_value = 1
            for k in range(num_layers * 2): t_value *= values[k][t_solution[k]]

            while True:
                b_idx, b_add_w, b_add_v = improve(t_solution, t_weight, [i, j])
                if b_idx == -1:
                    break
                t_solution[b_idx] += 1
                t_weight += b_add_w
                t_value *= b_add_v

            if t_value > best_value:
                f_solution = t_solution
                best_value = t_value
                break

        if best_value == prev_best_value:
            step_size += 1
            if step_size > max_step_size: break
            continue

        bpw = t_weight / numel
        print(f" -- Score: {best_value:.8f}  bpw: {bpw:.4f}")
        prev_best_value = best_value

    # Save strategy

    print(" -- Quantization strategy:")

    errp = 1
    job["strategy"] = {}
    for layer_ in range(num_layers):

        k1 = "model.layers." + str(layer_) + ".self_attn"
        k2 = "model.layers." + str(layer_) + "." + mlp_mode
        p1 = params[layer_ * 2][f_solution[layer_ * 2]]
        p2 = params[layer_ * 2 + 1][f_solution[layer_ * 2 + 1]]

        for (k, p, n) in zip((k1, k2), (p1, p2), (numel_attn, numel_mlp)):
            job["strategy"][k] = p
            bpw = p["total_bits"] / n
            err = 1 - p["accuracy"]
            print(f" --   {k:50} {bpw:1.4f} bpw - exp. error: {err:1.8f}")
            errp *= (1 - err)

    print(f" -- Total exp. error: {1 - errp:1.12f}")

    xx = 0