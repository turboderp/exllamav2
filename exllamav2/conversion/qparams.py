import math

class QParams:

    group_size: dict
    bits: list
    bits_prop: list
    scale_bits: int

    desc: str

    def __init__(self, group_size, bits, bits_prop, scale_bits):

        self.bits = bits
        self.bits_prop = bits_prop
        self.scale_bits = scale_bits

        # Allow group size per bitrate

        if isinstance(group_size, dict):
            self.group_size = { int(b): g for b, g in group_size.items() }
        elif isinstance(group_size, list):
            assert len(group_size) == len(bits)
            self.group_size = { b: g for g, b in zip(group_size, bits) }
        else:
            self.group_size = { b: group_size for b in bits }

        self.desc = self.get_desc()


    def __repr__(self):

        if len(set(self.group_size.values())) == 1:
            _gs = str(list(self.group_size.values())[0])
        else:
            _gs = "[" + ", ".join(str(self.group_size[b]) for b in self.bits) + "]"
        _b = "[" + ", ".join(str(b) for b in self.bits) + "]"
        _bp = "[" + ", ".join(str(bp) for bp in self.bits_prop) + "]"
        _sb = "4"
        return "QParams(" + _gs + ", " + _b + ", " + _bp + ", " + _sb + ")"


    def get_dict(self):

        return { "group_size": self.group_size,
                 "bits": self.bits,
                 "bits_prop": self.bits_prop,
                 "scale_bits": self.scale_bits }


    @staticmethod
    def from_dict(qp_dict):

        return QParams(qp_dict["group_size"],
                       qp_dict["bits"],
                       qp_dict["bits_prop"],
                       qp_dict["scale_bits"])


    def total_bits(self, shape, bias_shape = None):

        rows = shape[0]
        columns = shape[1]
        numel = rows * columns

        groups = 0
        remaining_rows = rows
        bits_groups = []
        for b, p in zip(self.bits, self.bits_prop):
            gsz = self.group_size[b]
            g = math.ceil(min(rows * p, remaining_rows) / gsz)
            groups += g
            remaining_rows -= g * gsz
            bits_groups.append(g)

        assert remaining_rows <= 0

        total_bits = 0
        tr = rows

        for g, b in zip(bits_groups, self.bits):

            r = self.group_size[b] * g
            c = columns
            if r > tr: r = tr
            tr -= r
            total_bits += r * c * b                         # q_weight

        total_bits += groups * 16                           # q_scale_max
        total_bits += groups * (16 + 16)                    # q_groups
        total_bits += groups * columns * self.scale_bits    # q_scale
        total_bits += rows * 32                             # q_invperm

        if bias_shape is not None:
            bias_numel = 1
            for d in bias_shape: bias_numel *= d
            total_bits += 16 * d

        return total_bits


    def bpw(self, shape, bias_shape = None):

        rows = shape[0]
        columns = shape[1]
        numel = rows * columns

        if bias_shape is not None:
            bias_numel = 1
            for d in bias_shape: bias_numel *= d
            numel += d

        return self.total_bits(shape, bias_shape) / numel


    def get_desc(self, filename = False):

        s = ""
        for b, p in zip(self.bits, self.bits_prop):
            if s != "": s += ("__" if filename else "/")
            g = self.group_size[b]
            s += f"{p}{'___' if filename else ':'}{b}b_{g}g"

        s += f" s{self.scale_bits}"

        return s


# kernels require groupsize divisible by 32, except 2-bit groups which can be divisible by 16

qparams_attn = \
[
    [
        QParams(64, [3, 2], [0.05, 0.95], 4),
        QParams(64, [3, 2], [0.05, 0.95], 4),
        QParams(64, [3, 2], [0.05, 0.95], 4),
        QParams(64, [3, 2], [0.05, 0.95], 4),
    ],
    [
        QParams(64, [3, 2], [0.1, 0.9], 4),
        QParams(64, [3, 2], [0.1, 0.9], 4),
        QParams(64, [3, 2], [0.25, 0.75], 4),
        QParams(64, [3, 2], [0.1, 0.9], 4),
    ],
    [
        QParams(64, [3, 2], [0.1, 0.9], 4),
        QParams(64, [3, 2], [0.1, 0.9], 4),
        QParams(128, [4, 3], [0.1, 0.9], 4),
        QParams(64, [3, 2], [0.1, 0.9], 4),
    ],
    [
        QParams(64, [3, 2], [0.1, 0.9], 4),
        QParams(64, [3, 2], [0.1, 0.9], 4),
        QParams(128, [4, 3], [0.1, 0.9], 4),
        QParams(128, [4, 3], [0.1, 0.9], 4),
    ],
    [
        QParams(128, [4, 3], [0.1, 0.9], 4),
        QParams(128, [4, 3], [0.1, 0.9], 4),
        QParams(128, [4, 3], [0.1, 0.9], 4),
        QParams(128, [4, 3], [0.1, 0.9], 4),
    ],
    [
        QParams(128, [4, 3], [0.1, 0.9], 4),
        QParams(128, [4, 3], [0.1, 0.9], 4),
        QParams(64, [4, 3], [0.1, 0.9], 4),
        QParams(128, [4, 3], [0.1, 0.9], 4),
    ],
    [
        QParams(128, [4], [1], 4),
        QParams(128, [4], [1], 4),
        QParams(128, [4], [1], 4),
        QParams(128, [4], [1], 4),
    ],
    [
        QParams(128, [4], [1], 4),
        QParams(128, [4], [1], 4),
        QParams(64, [4], [1], 4),
        QParams(128, [4], [1], 4),
    ],
    [
        QParams(64, [4], [1], 4),
        QParams(64, [4], [1], 4),
        QParams(32, [4], [1], 4),
        QParams(64, [4], [1], 4),
    ],
    [
        QParams(32, [4], [1], 4),
        QParams(32, [4], [1], 4),
        QParams(32, [4], [1], 4),
        QParams(32, [4], [1], 4),
    ],
    [
        QParams(128, [5, 4], [0.1, 0.9], 4),
        QParams(128, [5, 4], [0.1, 0.9], 4),
        QParams(64, [5, 4], [0.1, 0.9], 4),
        QParams(128, [5, 4], [0.1, 0.9], 4),
    ],
    [
        QParams(64, [5, 4], [0.1, 0.9], 4),
        QParams(64, [5, 4], [0.1, 0.9], 4),
        QParams(32, [5, 4], [0.1, 0.9], 4),
        QParams(64, [5, 4], [0.1, 0.9], 4),
    ],
    [
        QParams(64, [5, 4], [0.1, 0.9], 4),
        QParams(64, [5, 4], [0.1, 0.9], 4),
        QParams(64, [5], [1], 4),
        QParams(64, [5, 4], [0.1, 0.9], 4),
    ],
    [
        QParams(32, [5, 4], [0.1, 0.9], 4),
        QParams(32, [5, 4], [0.1, 0.9], 4),
        QParams(32, [5], [1], 4),
        QParams(32, [5, 4], [0.1, 0.9], 4),
    ],
    [
        QParams(128, [6, 5], [0.1, 0.9], 4),
        QParams(128, [6, 5], [0.1, 0.9], 4),
        QParams(128, [6], [1], 4),
        QParams(128, [6, 5], [0.1, 0.9], 4),
    ],
    [
        QParams(32, [6, 5], [0.1, 0.9], 4),
        QParams(32, [6, 5], [0.1, 0.9], 4),
        QParams(32, [6], [1], 4),
        QParams(32, [6, 5], [0.1, 0.9], 4),
    ],
    [
        QParams(128, [6], [1], 4),
        QParams(128, [6], [1], 4),
        QParams(128, [6], [1], 4),
        QParams(128, [6], [1], 4),
    ],
    [
        QParams(32, [6], [1], 4),
        QParams(32, [6], [1], 4),
        QParams(32, [8], [1], 4),
        QParams(32, [6], [1], 4),
    ],
    [
        QParams(128, [8], [1], 4),
        QParams(128, [8], [1], 4),
        QParams(128, [8], [1], 4),
        QParams(128, [8], [1], 4),
    ],
]

qparams_mlp = \
[
    [
        QParams(64, [3, 2], [0.05, 0.95], 4),
        QParams(64, [3, 2], [0.05, 0.95], 4),
        QParams([32, 64, 64], [6, 3, 2], [0.05, 0.2, 0.75], 4),
    ],
    [
        QParams(64, [3, 2], [0.1, 0.9], 4),
        QParams(64, [3, 2], [0.25, 0.75], 4),
        QParams([32, 64, 64], [6, 3, 2], [0.05, 0.2, 0.75], 4),
    ],
    [
        QParams(64, [3, 2], [0.1, 0.9], 4),
        QParams(64, [3, 2], [0.3, 0.7], 4),
        QParams(32, [5, 3], [0.05, 0.95], 4),
    ],
    [
        QParams(64, [3, 2], [0.1, 0.9], 4),
        QParams(64, [3, 2], [0.3, 0.7], 4),
        QParams(32, [5, 4], [0.05, 0.95], 4),
    ],
    [
        QParams(128, [4, 3], [0.1, 0.9], 4),
        QParams(128, [4, 3], [0.25, 0.75], 4),
        QParams([32, 128, 128], [8, 4, 3], [0.05, 0.1, 0.85], 4),
    ],
    [
        QParams(32, [4, 3], [0.1, 0.9], 4),
        QParams(32, [4, 3], [0.25, 0.75], 4),
        QParams([32, 32, 32], [8, 4, 3], [0.05, 0.1, 0.85], 4),
    ],
    [
        QParams(32, [4, 3], [0.1, 0.9], 4),
        QParams(32, [4, 3], [0.25, 0.75], 4),
        QParams([32, 128], [8, 4], [0.05, 0.95], 4),
    ],
    [
        QParams(128, [4], [1], 4),
        QParams(32, [4], [1], 4),
        QParams([32, 128], [8, 4], [0.05, 0.95], 4),
    ],
    [
        QParams(32, [4], [1], 4),
        QParams(32, [4], [1], 4),
        QParams([32, 32], [8, 4], [0.05, 0.95], 4),
    ],
    [
        QParams(128, [5, 4], [0.1, 0.9], 4),
        QParams(128, [5, 4], [0.25, 0.75], 4),
        QParams([32, 128, 128], [8, 5, 4], [0.05, 0.1, 0.85], 4),
    ],
    [
        QParams(32, [5, 4], [0.1, 0.9], 4),
        QParams(32, [5, 4], [0.25, 0.75], 4),
        QParams([32, 32, 32], [8, 5, 4], [0.05, 0.1, 0.85], 4),
    ],
    [
        QParams(128, [6, 5], [0.1, 0.9], 4),
        QParams(128, [6, 5], [0.25, 0.75], 4),
        QParams([32, 128, 128], [8, 6, 5], [0.05, 0.1, 0.85], 4),
    ],
    [
        QParams(32, [6, 5], [0.1, 0.9], 4),
        QParams(32, [6, 5], [0.25, 0.75], 4),
        QParams([32, 32, 32], [8, 6, 5], [0.05, 0.1, 0.85], 4),
    ],
    [
        QParams(128, [6], [1], 4),
        QParams(128, [6], [1], 4),
        QParams([32, 128], [8, 6], [0.05, 0.95], 4),
    ],
    [
        QParams(128, [8, 6], [0.1, 0.9], 4),
        QParams(128, [8, 6], [0.1, 0.9], 4),
        QParams(128, [8, 6], [0.15, 0.85], 4),
    ],
    [
        QParams(128, [8, 6], [0.1, 0.9], 4),
        QParams(128, [8, 6], [0.1, 0.9], 4),
        QParams(128, [8], [1], 4),
    ],
    [
        QParams(128, [8], [1], 4),
        QParams(128, [8], [1], 4),
        QParams(128, [8], [1], 4),
    ]
]

qparams_headoptions = \
{
    2: QParams(32, [4, 2], [0.3, 0.7], 4),
    3: QParams(32, [4, 3], [0.15, 0.85], 4),
    4: QParams(32, [6, 4], [0.15, 0.85], 4),
    5: QParams(128, [6, 5], [0.15, 0.85], 4),
    6: QParams(128, [8, 6], [0.15, 0.85], 4),
    8: QParams(128, [8], [1.0], 4),
    # 16: None
}

def get_qparams_reduced(options, ignore_gate = False):

    num_options = len(options)
    dim = len(options[0])
    assert all(len(o) == dim for o in options)

    desc_to_idx = [{} for _ in range(dim)]
    idx_to_qp = [[] for _ in range(dim)]
    maps = []

    for o in options:
        m = []
        for idx, qp in enumerate(o):
            if ignore_gate and idx == 0: continue
            desc = qp.get_desc()
            if desc not in desc_to_idx[idx]:
                j = len(idx_to_qp[idx])
                desc_to_idx[idx][desc] = j
                idx_to_qp[idx].append(qp)
            else:
                j = desc_to_idx[idx][desc]
            m.append(j)
        maps.append(m)

    return idx_to_qp, maps
