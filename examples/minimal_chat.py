from exllamav2 import *
from exllamav2.generator import *
import sys, torch

print("Loading model...")

config = ExLlamaV2Config("/mnt/str/models/mixtral-8x7b-instruct-exl2/3.0bpw/")
model = ExLlamaV2(config)
cache = ExLlamaV2Cache(model, lazy = True)
model.load_autosplit(cache)

tokenizer = ExLlamaV2Tokenizer(config)
generator = ExLlamaV2DynamicGenerator(model, cache, tokenizer)
context_ids = torch.empty((1, 0), dtype = torch.long)

while True:

    print()
    instruction = input("User: ")
    print()
    print("Assistant:", end = "")

    instruction_ids = tokenizer.encode(f"[INST] {instruction} [/INST]", add_bos = True)
    context_ids = torch.cat([context_ids, instruction_ids], dim = -1)

    generator.enqueue(
        ExLlamaV2DynamicJob(
            input_ids = context_ids,
            max_new_tokens = 1024,
            stop_conditions = [tokenizer.eos_token_id],
        )
    )

    eos = False
    while not eos:
        results = generator.iterate()
        for result in results:
            if result["stage"] == "streaming":
                eos = result["eos"]
                if "text" in result:
                    print(result["text"], end = "")
                    sys.stdout.flush()
                if "token_ids" in result:
                    context_ids = torch.cat([context_ids, result["token_ids"]], dim = -1)

    print()
