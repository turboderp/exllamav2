from . import latticed4, latticee8_padded12

# name: (id, codebook class)
# codebook_id = {
#     'D4': (0, latticed4.D4_codebook),
#     'E8P12': (7, latticee8_padded12.E8P12_codebook),
#     # 'HI4B1C': (10, half_integer_4bit_1col.HI4B1C_codebook),
# }

# id from above:6quantized linear implementation
quantized_class = {
    0: latticed4.QuantizedD4Linear,
    7: latticee8_padded12.QuantizedE8P12Linear,
    # 10: half_integer_4bit_1col.QuantizedHI4B1CLinear,
}

# cache_permute_set = {
#     0,  # D4
# }


# def get_codebook(name):
#     return codebook_id[name][1]()


# def get_id(name):
#     return codebook_id[name][0]


def get_quantized_class(id):
    return quantized_class[id]
