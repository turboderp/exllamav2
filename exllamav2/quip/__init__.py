from . import latticed4, latticee8_padded12, half_integer_4bit_1col

quantized_class = {
    0: latticed4.QuantizedD4Linear,
    7: latticee8_padded12.QuantizedE8P12Linear,
    10: half_integer_4bit_1col.QuantizedHI4B1CLinear,
}

def get_quantized_class(id):
    return quantized_class[id]
