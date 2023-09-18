
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
