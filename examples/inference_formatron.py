
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator
from formatron.schemas.pydantic import ClassSchema
from formatron.integrations.exllamav2 import create_formatter_filter
from formatron.formatter import FormatterBuilder
# from pydantic import conlist
from typing import Literal, Optional
import json
import argparse

def create_superhero_json_formatter():

    class SuperheroAppearance(ClassSchema):
        title: str
        issue_number: int
        year: int

    class Superhero(ClassSchema):
        name: str
        secret_identity: str
        gender: Literal["male", "female"]
        # superpowers: conlist(str, max_length = 5)  # conlist is not currently supported by Formatron
        first_appearance: SuperheroAppearance

    f = FormatterBuilder()
    f.append_line(f"{f.json(Superhero, capture_name='json')}")
    return f


def generate_json(model, tokenizer, generator):

    # Prepare prompts, alternating between raw and filtered

    i_prompts = [
        "Here is some information about Superman:\n\n",
        "Here is some information about Batman:\n\n",
        "Here is some information about Aquaman:\n\n",
    ]

    formatter = create_superhero_json_formatter()

    prompts = []
    filters = []

    for p in i_prompts:
        prompts.append(p)
        filters.append(None)
        prompts.append(p)
        filters.append([
            create_formatter_filter(model, tokenizer, formatter),
        ])

    # Generate

    print("Generating JSON examples...")

    outputs = generator.generate(
        prompt = prompts,
        filters = filters,
        max_new_tokens = 300,
        add_bos = True,
        stop_conditions = [tokenizer.eos_token_id],
        completion_only = True
    )

    # Print outputs:

    for i in range(len(i_prompts)):
        print("---------------------------------------------------------------------------------")
        print(i_prompts[i].strip())
        print()
        print("Without filter:")
        print("---------------")
        print(outputs[i * 2])
        print()
        print("With filter:")
        print("------------")
        print(json.dumps(json.loads(outputs[i * 2 + 1]), indent=4).strip())
        print()


def create_benchmark_json_formatter():

    occupations = [
        "Accountant",
        "Actor",
        "Acupuncturist",
        "Acute Care Nurse",
        "Adapted Physical Education Specialist",
        "Adhesive Bonding Machine Operator",
        "Administrative Law Judge",
        "Administrative Services Manager",
        "Advanced Practice Psychiatric Nurse",
        "Advertising Sales Agent",
        "Aerospace Engineer",
        "Cat"
    ]

    class Address(ClassSchema):
        street: str
        city: str
        state: Optional[str] = None
        postal_code: str
        country: str

    class Person(ClassSchema):
        full_name: str
        initials: str
        occupation: Literal[*occupations]
        previous_occupation: Literal[*occupations]
        work_address: Address
        home_address: Address
        cat_owner: bool
        dog_owner: bool
        nationality: str

    f = FormatterBuilder()
    f.append_line("\n\nParsed information, JSON format:\n")
    f.append_line(f"{f.json(Person, capture_name = 'json')}")
    return f


def test_overhead(model, tokenizer, generator, bsz):

    prompts = [
        """Emily Jane Parker is currently working as an Administrative Services Manager at Greenfield Solutions. Before """
        """this, she was an Advertising Sales Agent. Her office is located at 789 Corporate Drive, Los Angeles, CA 90210, """
        """USA. She resides at 321 Oak Street, Pasadena, CA 91103, USA. Emily owns several cats who often keep her company. """
        """She is a proud citizen of the United States."""
    ] * bsz

    # Warm up at bsz to make sure kernels are tuned and graphs are built

    print(f"bsz {bsz}, warmup...")

    outputs = generator.generate(
        prompt = prompts,
        min_new_tokens = 499,
        max_new_tokens = 500,
        add_bos = True,
        stop_conditions = [tokenizer.eos_token_id],
        completion_only = True,
    )

    # Generate at bsz

    print(f"bsz {bsz}, constrained generation...")

    formatter = create_benchmark_json_formatter()
    filters = [[create_formatter_filter(model, tokenizer, formatter)] for _ in range(bsz)]

    outputs, res = generator.generate(
        prompt = prompts,
        filters = filters,
        max_new_tokens = 500,
        add_bos = True,
        stop_conditions = [tokenizer.eos_token_id],
        completion_only = True,
        return_last_results = True
    )

    avg_tokens = sum(r["new_tokens"] for r in res) / bsz
    avg_time = sum(r["time_generate"] for r in res) / bsz
    tps_filtered = avg_tokens / avg_time
    print(f"bsz {bsz}, constrained generation: {tps_filtered:.2f} t/s")

    # Decode JSON and print to the console to verify that filter is working

    j = outputs[0]
    j = j[j.find('{'):]
    j = json.dumps(json.loads(j), indent = 4).strip()
    print(j)

    # Generate at bsz for refernece, unfiltered

    print(f"bsz {bsz}, unconstrained generation...")

    outputs, res = generator.generate(
        prompt = prompts,
        max_new_tokens = int(avg_tokens),
        add_bos = True,
        stop_conditions = [tokenizer.eos_token_id],
        completion_only = True,
        return_last_results = True
    )

    avg_tokens = sum(r["new_tokens"] for r in res) / bsz
    avg_time = sum(r["time_generate"] for r in res) / bsz
    tps_unfiltered = avg_tokens / avg_time
    print(f"bsz {bsz}, unconstrained generation: {tps_unfiltered:.2f} t/s")

    return tps_filtered, tps_unfiltered


def main(benchmark: bool):

    # Load model etc.

    model_dir = "/mnt/str/models/llama3-8b-exl2/6.0bpw"
    config = ExLlamaV2Config(model_dir)
    config.arch_compat_overrides()
    model = ExLlamaV2(config)
    cache = ExLlamaV2Cache(model, max_seq_len = 32768, lazy = True)
    model.load_autosplit(cache, progress = True)

    print("Loading tokenizer...")
    tokenizer = ExLlamaV2Tokenizer(config)

    # Initialize the generator with all default parameters

    generator = ExLlamaV2DynamicGenerator(
        model = model,
        cache = cache,
        tokenizer = tokenizer,
    )

    # Benchmark

    if benchmark:

        batchsizes = [1, 2, 4]
        results = []

        for bsz in batchsizes:
            tps_filtered, tps_unfiltered = test_overhead(model, tokenizer, generator, bsz)
            overhead = 1 - tps_filtered / tps_unfiltered
            latency = 1 / tps_filtered - 1 / tps_unfiltered
            results.append((overhead, latency))

        print("---------------------------------------------------------------------------------")
        print("Results:")
        print()
        for bsz, (overhead, latency) in zip(batchsizes, results):
            print(f"bsz {bsz}: overhead {overhead*100:8.2f}%     latency: {latency*1000:8.4f} ms")

    # Run example generation

    else:
        generate_json(model, tokenizer, generator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--benchmark", action = "store_true", help = "Run benchmark")
    args = parser.parse_args()
    main(args.benchmark)
