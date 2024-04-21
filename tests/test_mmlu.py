
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
import hashlib
import json

# Models to test

# model_base = "/mnt/str/models/_exl2"
# model_base = "/mnt/str/models/mixtral-8x7b-instruct-exl2/"
model_base = "/mnt/str/models/llama3-8b-exl2"
# variants = ["x3-8b"]

variants = [v for v in os.listdir(model_base) if os.path.isdir(os.path.join(model_base, v))]

# variants = \
# [
#     "2.4bpw",
#     "2.5bpw",
#     "3.0bpw",
#     "4.0bpw",
#     "6.0bpw",
# ]

gpu_split = (20, 21.3, 24)

qa_set = "cais/mmlu"
qa_split = "test"

categories = \
[
    "anatomy",
    "computer_security",
    "formal_logic",
    "logical_fallacies",
    "philosophy",
    "nutrition",
]

examples_per_category = 3
questions_per_category = 97

# Load model

def get_model(base, variant_, gpu_split_, batch_size_):

    model_dir = os.path.join(base, variant_)

    config = ExLlamaV2Config()
    config.model_dir = model_dir
    config.prepare()
    config.max_seq_len = 2048
    config.max_batch_size = batch_size_

    model_ = ExLlamaV2(config)
    print(" -- Loading model: " + model_dir)

    model_.load(gpu_split_)

    tokenizer_ = ExLlamaV2Tokenizer(config)

    # cache_ = ExLlamaV2Cache(model)
    cache_ = None

    return model_, cache_, tokenizer_



# Prepare the prompts

def load_datasets():
    global categoriwes

    def get_dataset(ds_name, category_, split_):

        print(f" -- Loading dataset: {ds_name}/{category_}...")
        dataset_ = load_dataset(ds_name, category_, split = split_)
        return dataset_

    def format_question(question, options, answer, ex=False):

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

    prep_prompts_ = {}
    for category_ in categories:

        dataset = get_dataset(qa_set, category_, qa_split)

        rows = []
        for example in dataset:
            rows.append(example)
            if len(rows) == questions_per_category + examples_per_category: break

        examples_prompt = ""
        for i_ in range(examples_per_category):
            examples_prompt += format_question(rows[i_]["question"], rows[i_]["choices"], rows[i_]["answer"], ex = True)
            examples_prompt += "\n\n"

        prompts_ = []
        labels_ = []
        for j_ in range(questions_per_category):
            i_ = j_ + examples_per_category
            q_prompt = format_question(rows[i_]["question"], rows[i_]["choices"], rows[i_]["answer"])
            prompts_.append(examples_prompt + q_prompt)
            labels_.append(rows[i_]["answer"])

        prep = {"prompts": prompts_,
                "labels": labels_}

        prep_prompts_[category_] = prep

    return prep_prompts_


hash_obj = hashlib.sha256()
hash_obj.update(''.join(categories).encode())
h = hash_obj.hexdigest()
module_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(module_dir, "mmlu_prompts_" + h[:12] + ".json")

if os.path.exists(filename):
    with open(filename, "r") as f:
        prep_prompts = json.load(f)
else:
    prep_prompts = load_datasets()
    with open(filename, "w") as f:
        f.write(json.dumps(prep_prompts, indent = 4))

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
        answer_logits.append(tokenizer.tokenizer.encode(answer_)[-1])

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
