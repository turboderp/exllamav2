
class QParams:

    group_size: int
    bits: list
    bits_prop: list
    scale_bits: int

    desc: str

    def __init__(self, group_size, bits, bits_prop, scale_bits):

        self.group_size = group_size
        self.bits = bits
        self.bits_prop = bits_prop
        self.scale_bits = scale_bits
        self.desc = self.get_desc()


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


    def total_bits(self, shape):

        rows = shape[0]
        columns = shape[1]
        numel = rows * columns

        groups = (rows + self.group_size - 1) // self.group_size

        g128 = (rows + 128 - 1) // 128
        bits_groups = [max(round(g128 * p), 1) * 128 // self.group_size for p in self.bits_prop]
        e = sum(bits_groups) - groups
        bits_groups[-1] -= e

        total_bits = 0
        tr = rows

        for g, b in zip(bits_groups, self.bits):

            r = self.group_size * g
            c = columns
            if r > tr: r = tr
            tr -= r
            total_bits += r * c * b                         # q_weight

        total_bits += groups * 16                           # q_scale_max
        total_bits += groups * (16 + 16)                    # q_groups
        total_bits += groups * columns * self.scale_bits    # q_scale
        total_bits ++ rows * 32                             # q_invperm

        return total_bits


    def bpw(self, shape):

        rows = shape[0]
        columns = shape[1]
        numel = rows * columns

        return self.total_bits(shape) / numel


    def get_desc(self):

        s = ""
        for b, p in zip(self.bits, self.bits_prop):
            if s != "": s += "/"
            s += f"{p}:{b}b"

        s += f" {self.group_size}g s{self.scale_bits}"

        return s


# kernels require groupsize divisible by 32

qparams_options = \
[
    QParams(32, [3, 2], [0.05, 0.95], 4),
    QParams(32, [3, 2], [0.25, 0.75], 4),
    QParams(32, [4, 2], [0.25, 0.75], 4),
    QParams(32, [4, 3, 2], [0.1, 0.4, 0.5], 4),
    QParams(32, [4, 3], [0.1, 0.9], 4),
    QParams(32, [6, 3], [0.2, 0.8], 4),
    QParams(128, [3], [1.0], 4),
    QParams(32, [3], [1.0], 4),
    QParams(32, [4, 3], [0.05, 0.95], 4),
    QParams(32, [4, 3], [0.4, 0.6], 4),
    QParams(64, [4, 3], [0.6, 0.4], 4),
    QParams(128, [4], [1.0], 4),
    QParams(32, [4], [1.0], 4),
    QParams(32, [5, 4], [0.1, 0.9], 4),
    QParams(32, [6, 4], [0.1, 0.9], 4),
    QParams(128, [5], [1.0], 4),
    QParams(32, [6, 5], [0.1, 0.9], 4),
    QParams(32, [8, 6, 5], [0.05, 0.05, 0.9], 4),
    QParams(32, [6, 5], [0.4, 0.6], 4),
    QParams(32, [8, 6, 5], [0.1, 0.3, 0.6], 4),
    QParams(128, [6], [1.0], 4),
    QParams(32, [6], [1.0], 4),
    QParams(128, [8, 6], [0.1, 0.9], 4),
    QParams(32, [8], [1.0], 4),
]

qparams_headoptions = \
{
    2: QParams(32, [4, 2], [0.3, 0.7], 4),
    3: QParams(32, [4, 3], [0.15, 0.85], 4),
    4: QParams(32, [6, 4], [0.15, 0.85], 4),
    5: QParams(32, [6, 5], [0.15, 0.85], 4),
    6: QParams(32, [8, 6], [0.15, 0.85], 4),
    8: QParams(32, [8], [1.0], 4),
    16: None
}
