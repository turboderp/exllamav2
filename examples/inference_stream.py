
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2DynamicJob
import pprint

model_dir = "/mnt/str/models/mistral-7b-exl2/4.0bpw"
config = ExLlamaV2Config(model_dir)
model = ExLlamaV2(config)
cache = ExLlamaV2Cache(model, lazy = True)
model.load_autosplit(cache, progress = True)

print("Loading tokenizer...")
tokenizer = ExLlamaV2Tokenizer(config)

# Initialize the generator with all default parameters

generator = ExLlamaV2DynamicGenerator(
    model = model,
    cache = cache,
    tokenizer = tokenizer,
)

# Start a generation job. We can add a number of arguments here like stop conditions, sample settings and more, but
# for this demonstration we'll only enable token healing, which addresses the extraneous space at the end of the
# prompt.

prompt = "Our story begins in the Scottish town of Auchtermuchty, where "
input_ids = tokenizer.encode(prompt, add_bos = False)
job = ExLlamaV2DynamicJob(
    input_ids = input_ids,
    max_new_tokens = 200,
    token_healing = True
)

generator.enqueue(job)

# Stream output to the terminal

print()
print(prompt, end = ""); sys.stdout.flush()

eos = False
while not eos:

    # Run one iteration of the generator. Returns a list of results
    results = generator.iterate()

    for result in results:

        # If we enqueue multiple jobs, an iteration might produce results for any (or all) of them. We could direct
        # outputs to multiple clients here, using whatever dispatch mechanism, but in this example there will only be
        # outputs pertaining to the single job started above, and it will all go straight to the console.
        assert result["job"] == job

        # Prefilling/ingesting the prompt may happen over multiple iterations, during which the result will have
        # a "stage" value of "prefill". We can ignore those results and only use the "streaming" results that will
        # contain the actual output.
        if result["stage"] == "streaming":

            # Depending on settings, the result dict can contain top-K probabilities, logits and more, but we'll just
            # grab the output text stream.
            text = result.get("text", "")
            print(text, end = ""); sys.stdout.flush()

            # The "streaming" stage also emits the EOS signal when it occurs. If present, it will accompany a
            # summary of the job. Print the last packet here to illustrate.
            if result["eos"]:
                print()
                print()
                print("---------------------------------")
                print("Generation complete. Last result:")
                print()
                pprint.pprint(result, indent = 4)
                eos = True