
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    # ExLlamaV2Cache,
    # ExLlamaV2Tokenizer,
)

import torch, time

# model_directory = "/mnt/str/models/_gptq/llama-7b-4bit-128g/"
# model_directory = "/mnt/str/models/llama2-7b-exl2/4.0bpw"
model_directory = "/mnt/str/models/_gptq/TheBloke_Phine-CodeLlama-34B-v2-GPTQ/"
# model_directory = "/mnt/str/models/llama2-7b"

config = ExLlamaV2Config()
config.model_dir = model_directory
config.prepare()

model = ExLlamaV2(config)
print("Loading model: " + model_directory)

model.load(gpu_split = [20, 20, 24])

samples = 100
samples_1 = 200
tests = list(range(1, 17)) + [20, 24, 28, 32] #, 48, 64, 96, 128, 256]

with torch.no_grad():
    for i in tests:
        input_ids = torch.randint(config.vocab_size - 1, (1, i))

        a = time.time()
        s = samples if i > 1 else samples_1
        for j in range(s):
            model.forward(input_ids)
        b = time.time()

        latency = (b - a) / s * 1000
        latency_tok = latency / i
        if i == 1: base_latency = latency
        efficiency = base_latency / latency

        print(f"{i:3} tokens     avg. latency: {latency:7.2f} ms     avg. latency/token: {latency_tok:7.2f} ms    batch eff.: {efficiency:.4f}")



