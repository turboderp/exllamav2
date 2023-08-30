
def optimize(job, save_fn):

    numel = 0
    max_rfn = 0.0
    for layer in job["measurement"]:
        numel += layer["numel"]
        for option in layer["options"]:
            max_rfn = max(max_rfn, option["err"])

    # Find minimum error that satisfies target size

    target_bpw = job["bits"]
    target_rfn = max_rfn

    target_rfn_step = max_rfn / 2

    while True:

        current_total_bits = 0
        for layer in job["measurement"]:

            best_option = None
            best_bpw = 100.0

            for option in layer["options"]:
                if option["bpw"] < best_bpw and option["err"] <= target_rfn:
                    best_bpw = option["bpw"]
                    best_option = option

            if best_option is not None:
                layer["best_option"] = best_option

            current_total_bits += int(layer["best_option"]["total_bits"])

        current_bpw = current_total_bits / numel
        print(f" -- rfn_error: {target_rfn:2.5f}  bpw: {current_bpw:2.5f}")

        eps = 1e-3

        if current_bpw < target_bpw - eps:
            target_rfn -= target_rfn_step
        elif  current_bpw > target_bpw + eps:
            target_rfn += target_rfn_step

        target_rfn_step /= 2
        if target_rfn_step < 1e-6: break

