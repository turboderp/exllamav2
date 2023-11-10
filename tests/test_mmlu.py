
import sys, os, gc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
)

from datasets import load_dataset
import torch

# Models to test

#model_base = "/mnt/str/models/_exl2"

model_base = "/mnt/str/models/_gptq/TheBloke_Yi-34B-GPTQ/"

# variants = [v for v in os.listdir(model_base) if os.path.isdir(os.path.join(model_base, v))]

variants = \
[
    "",
    # "goliath-120b-exl2/3.0bpw",
    # "llama2-70b-exl2/3.0bpw",
    # "llama2-70b-exl2/4.65bpw",
]

gpu_split = (21.2, 24)

qa_set = "cais/mmlu"
qa_split = "test"

categories = \
[
    "anatomy",
    # "computer_security",
    # "formal_logic",
    # "logical_fallacies",
    # "computer_security",
    # "philosophy",
    # "nutrition",
]

examples_per_category = 3
questions_per_category = 97

# Load model

def get_model(base, variant_, gpu_split_, batch_size_):

    model_dir = os.path.join(base, variant_)

    config = ExLlamaV2Config()
    config.model_dir = model_dir
    config.prepare()
    config.max_batch_size = batch_size_

    model_ = ExLlamaV2(config)
    print(" -- Loading model: " + model_dir)

    model_.load(gpu_split_)

    tokenizer_ = ExLlamaV2Tokenizer(config)

    # cache_ = ExLlamaV2Cache(model)
    cache_ = None

    return model_, cache_, tokenizer_


# Load questions

def format_question(question, options, answer, ex = False):

    clabels = "ABCD"
    text = f"Question:\n"
    text += question
    text += "\n\nChoices:\n"
    for i, o in enumerate(options):
        text += clabels[i] + ": " + o + "\n"
    text += "\nAnswer: " + clabels[answer]
    # if ex:
    #     text += ", " + options[answer]
    return text


def get_dataset(ds_name, category_, split_):

    print(f" -- Loading dataset: {ds_name}/{category_}...")
    dataset_ = load_dataset(ds_name, category_, split = split_)
    return dataset_


# Prepare the prompts

prep_prompts = {}
for category in categories:

    dataset = get_dataset(qa_set, category, qa_split)

    rows = []
    for example in dataset:
        rows.append(example)
        if len(rows) == questions_per_category + examples_per_category: break

    examples_prompt = ""
    for i in range(examples_per_category):
        examples_prompt += format_question(rows[i]["question"], rows[i]["choices"], rows[i]["answer"], ex = True)
        examples_prompt += "\n\n"

    prompts = []
    labels = []
    for j in range(questions_per_category):
        i = j + examples_per_category
        q_prompt = format_question(rows[i]["question"], rows[i]["choices"], rows[i]["answer"])
        prompts.append(examples_prompt + q_prompt)
        labels.append(rows[i]["answer"])

    prep = {"prompts": prompts,
            "labels": labels}

    prep_prompts[category] = prep


# Do the test

results = ";".join([""] + categories) + "\n"

for variant in variants:

    # Model

    model = None
    cache = None
    tokenizer = None

    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    model, cache, tokenizer = get_model(model_base, variant, gpu_split, 1)

    # Logit positions corresponding to valid answers

    answer_logits = []
    llabels = "ABCD"
    for i in range(4):
        answer_ = "The answer is: " + llabels[i]
        answer_logits.append(tokenizer.tokenizer.EncodeAsIds(answer_)[-1])

    # Categories

    cat_results = []

    for category in categories:

        print(f" -- Testing: {category}...")

        prompts = prep_prompts[category]["prompts"]
        labels = prep_prompts[category]["labels"]

        # Evaluate prompts

        score = 0.0
        # for prompt_ids, mask in zip(prompt_ids_list, mask_list):

        for prompt, label in zip(prompts, labels):

            prompt_ids = tokenizer.encode(prompt)
            prompt_ids = prompt_ids[:, :-1]

            logits = model.forward(prompt_ids, last_id_only = True)
            logits = logits.float()

            logits_ans = logits[:, :, answer_logits]
            prob_ans = torch.softmax(logits_ans, dim = -1)

            score += prob_ans[0, 0, label]

        score /= questions_per_category
        print(f" -- Score: {score:.4f}")

        cat_results.append(f"{score:.4f}");

    results += ";".join([variant] + cat_results) + "\n"

print(" -- Finished")
print()
print(results)
