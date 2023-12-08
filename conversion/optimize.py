from conversion.qparams import QParams, qparams_options
from conversion.qparams_stats import qparams_stats
import math
import itertools

def optimize(job, save_fn):

    eps = 0.0001

    numel = 0
    max_rfn = 0.0
    for layer in job["measurement"]:
        numel += layer["numel"]
        for option in layer["options"]:
            max_rfn = max(max_rfn, option["err"])

    # max_rfn -= eps
    min_rfn = 0
    best_rfn = 10000.0
    target_bpw = job["bits"]

    # Binary search for combination of settings that minimizes max rfn_error while
    # TODO: Detect if bpw is too low to be attainable
    # TODO: Fix potential broken models when bpw is too high

    invalid = False
    min_diff = 0.00001
    while max_rfn - min_rfn > min_diff or invalid:

        target_rfn = (min_rfn + max_rfn) / 2

        invalid = False
        current_total_bits = 0
        for layer in job["measurement"]:

            best_option = None
            best_bpw = 10000.0

            for option in layer["options"]:
                if option["bpw"] < best_bpw and option["err"] <= target_rfn:
                    best_bpw = option["bpw"]
                    best_option = option

            layer["best_option_max"] = best_option
            if best_option is None:
                invalid = True
                break

            current_total_bits += int(layer["best_option_max"]["total_bits"])

        current_bpw = current_total_bits / numel

        if not invalid:
            print(f" -- rfn max: {target_rfn:2.5f}  bpw: {current_bpw:2.5f}")
        else:
            print(f" -- rfn max: {target_rfn:2.5f}  (not possible)")

        if current_bpw <= target_bpw and not invalid:
            best_rfn = min(best_rfn, target_rfn)
            max_rfn = target_rfn
        else:
            min_rfn = target_rfn
            max_rfn += eps

    # We've found the smallest error that can be met by _all_ layers while staying below the set no. bits.
    # Now select a minimum target to allow some layers to use more accurate settings if we didn't meet the
    # target bitrate

    max_rfn = max(target_rfn, best_rfn)
    min_rfn = 0

    min_diff = 0.00001
    while max_rfn - min_rfn > min_diff:

        target_rfn = (min_rfn + max_rfn) / 2
        invalid = False

        current_total_bits = 0
        for layer in job["measurement"]:

            best_option = None
            best_rfn = 10000.0

            for option in layer["options"]:
                if best_rfn > option["err"] >= target_rfn and option["err"] < layer["best_option_max"]["err"]:
                    best_rfn = option["err"]
                    best_option = option

            if best_option is None:
                layer["best_option"] = layer["best_option_max"]
            else:
                layer["best_option"] = best_option

            current_total_bits += int(layer["best_option"]["total_bits"])

        current_bpw = current_total_bits / numel

        print(f" -- rfn min: {target_rfn:2.5f}  bpw: {current_bpw:2.5f}")

        if current_bpw <= target_bpw:
            max_rfn = target_rfn
        else:
            min_rfn = target_rfn


def optimize_new(job, save_fn, model):

    key = "model.layers.0"
    key_q = key + ".self_attn.q_proj"
    key_k = key + ".self_attn.k_proj"
    key_v = key + ".self_attn.v_proj"
    key_o = key + ".self_attn.o_proj"
    key_g = key + ".mlp.gate_proj"
    key_u = key + ".mlp.up_proj"
    key_d = key + ".mlp.down_proj"
    shape_q = model.modules_dict[key_q].matrix_shape()
    shape_k = model.modules_dict[key_k].matrix_shape()
    shape_v = model.modules_dict[key_v].matrix_shape()
    shape_o = model.modules_dict[key_o].matrix_shape()
    shape_g = model.modules_dict[key_g].matrix_shape()
    shape_u = model.modules_dict[key_u].matrix_shape()
    shape_d = model.modules_dict[key_d].matrix_shape()
    numel_q = shape_q[0] * shape_q[1]
    numel_k = shape_k[0] * shape_k[1]
    numel_v = shape_v[0] * shape_v[1]
    numel_o = shape_o[0] * shape_o[1]
    numel_g = shape_g[0] * shape_g[1]
    numel_u = shape_u[0] * shape_u[1]
    numel_d = shape_d[0] * shape_d[1]

    num_layers = model.config.num_hidden_layers
    numel = num_layers * (numel_q + numel_k + numel_v + numel_o + numel_g + numel_u + numel_d)
    target_bpw = job["bits"]
    weight_budget = numel * target_bpw

    layer_p1 = num_layers // 2
    layer_p2 = num_layers * 3 // 4
    layer_p3 = num_layers - 1
    assert 2 < layer_p1 < layer_p2 < layer_p3

    # Now it's a knapsack problem all of a sudden

    weights = []
    values = []
    params = []
    for i in range(num_layers * 2):
        weights.append([])
        values.append([])
        params.append([])

    for qcosts in qparams_stats:
        mode_q, mode_k, mode_v, mode_o, mode_g, mode_u, mode_d = qcosts[:7]

        if mode_q:
            bits = 0
            bits += mode_q.total_bits(shape_q)
            bits += mode_k.total_bits(shape_k)
            bits += mode_v.total_bits(shape_v)
            bits += mode_o.total_bits(shape_o)
            index = 0

        else:
            bits = 0
            bits += mode_g.total_bits(shape_g)
            bits += mode_u.total_bits(shape_u)
            bits += mode_d.total_bits(shape_d)
            index = 1

        layer_kldiv = qcosts[7:]
        for layer in range(num_layers):
            if layer == 0:
                kldiv = layer_kldiv[0]
            elif layer == 1:
                kldiv = layer_kldiv[1]
            elif layer == 2:
                kldiv = layer_kldiv[2]
            elif layer < layer_p1:
                a = (layer_p1 - layer) / (layer_p1 - 2)
                b = (layer - 2) / (layer_p1 - 2)
                kldiv = a * layer_kldiv[2] + b * layer_kldiv[3]
            elif layer == layer_p1:
                kldiv = layer_kldiv[3]
            elif layer < layer_p2:
                a = (layer_p2 - layer) / (layer_p2 - layer_p1)
                b = (layer - layer_p1) / (layer_p2 - layer_p1)
                kldiv = a * layer_kldiv[3] + b * layer_kldiv[4]
            elif layer == layer_p2:
                kldiv = layer_kldiv[4]
            elif layer < layer_p3:
                a = (layer_p3 - layer) / (layer_p3 - layer_p2)
                b = (layer - layer_p2) / (layer_p3 - layer_p2)
                kldiv = a * layer_kldiv[4] + b * layer_kldiv[5]
            else:
                kldiv = layer_kldiv[5]

            weights[2 * layer + index].append(bits)
            values[2 * layer + index].append(kldiv)
            params[2 * layer + index].append(qcosts)

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
            if v_[j] >= v_[j - 1]:
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
    value = sum(values[i][0] for i in range(num_layers * 2))

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

            add_v = values[idx][si + 1] - values[idx][si]
            ratio = -add_v / add_w
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = idx
                best_add_w = add_w
                best_add_v = add_v

        return best_idx, best_add_w, best_add_v

    while True:
        b_idx, b_add_w, b_add_v = improve(f_solution, weight)
        if b_idx == -1:
            break

        f_solution[b_idx] += 1
        weight += b_add_w
        value += b_add_v

    bpw = weight / numel
    print(f" -- Estimated divergence: {value:.8f}  bpw: {bpw:.4f}")

    best_value = value
    prev_best_value = value
    step_size = 1

    while True:

        for i, j in itertools.permutations(range(num_layers * 2), 2):

            t_solution = f_solution.copy()
            t_solution[i] = max(t_solution[i] - step_size, 0)
            t_solution[j] = max(t_solution[j] - step_size, 0)

            t_weight = sum(weights[i][t_solution[i]] for i in range(num_layers * 2))
            t_value = sum(values[i][t_solution[i]] for i in range(num_layers * 2))

            while True:
                b_idx, b_add_w, b_add_v = improve(t_solution, t_weight, [i, j])
                if b_idx == -1:
                    break
                t_solution[b_idx] += 1
                t_weight += b_add_w
                t_value += b_add_v

            if t_value < best_value:
                f_solution = t_solution
                best_value = t_value
                break

        if best_value == prev_best_value:
            step_size += 1
            if step_size > 2: break
            continue

        bpw = t_weight / numel
        print(f" -- Estimated divergence: {best_value:.8f}  bpw: {bpw:.4f}")
        prev_best_value = best_value

    # Compile as measurement

    print(" -- Quantization strategy:")

    job["measurement"] = []
    for layer_ in range(num_layers):

        key = "model.layers." + str(layer_)
        key_q = key + ".self_attn.q_proj"
        key_k = key + ".self_attn.k_proj"
        key_v = key + ".self_attn.v_proj"
        key_o = key + ".self_attn.o_proj"
        key_g = key + ".mlp.gate_proj"
        key_u = key + ".mlp.up_proj"
        key_d = key + ".mlp.down_proj"

        qp1 = params[layer_ * 2][f_solution[layer_ * 2]]
        qp2 = params[layer_ * 2 + 1][f_solution[layer_ * 2 + 1]]
        mode_q, mode_k, mode_v, mode_o, _, _, _ = qp1[:7]
        _, _, _, _, mode_g, mode_u, mode_d = qp2[:7]

        def store_res(key_, numel_, mode_, shape_):
            bpw_ = mode_.bpw(shape_)
            desc_ = mode_.get_desc()
            print(f" --   {key_:40}  bpw: {bpw_:.4f}  mode: {desc_}")
            job["measurement"].append({
                "key": key_,
                "numel": numel_,
                "best_option": {
                    "desc": desc_,
                    "bpw": bpw_,
                    "total_bits": mode_.total_bits(shape_),
                    "err": 0,
                    "qparams": mode_.get_dict()
                }
            })

        store_res(key_q, numel_q, mode_q, shape_q)
        store_res(key_k, numel_k, mode_k, shape_k)
        store_res(key_v, numel_v, mode_v, shape_v)
        store_res(key_o, numel_o, mode_o, shape_o)
        store_res(key_g, numel_g, mode_g, shape_g)
        store_res(key_u, numel_u, mode_u, shape_u)
        store_res(key_d, numel_d, mode_d, shape_d)
