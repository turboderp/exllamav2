from __future__ import annotations
import sys, argparse, random, torch
import util
from exllamav2 import model_init, ExLlamaV2Cache, ExLlamaV2Cache_Q4, ExLlamaV2Cache_Q6, ExLlamaV2Cache_Q8
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2DynamicJob, ExLlamaV2Sampler
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
from collections import defaultdict

# Argument Parsing
parser = argparse.ArgumentParser(description="Run MMLU evaluation on EXL2 model")
parser.add_argument("-cs", "--cache_size", type=int, default=None)
parser.add_argument("-cq4", "--cache_q4", action="store_true", help="Use Q4 cache")
parser.add_argument("-cq6", "--cache_q6", action="store_true", help="Use Q6 cache")
parser.add_argument("-cq8", "--cache_q8", action="store_true", help="Use Q8 cache")
parser.add_argument("-sub", "--subjects", type=str, default="all", help="Comma-separated list of categories to test, or 'all'")
parser.add_argument("-fs", "--fewshot_examples", type=int, default=5, help="Number of examples for fewshot examples, max 5")
parser.add_argument("-shf", "--shuffle", action="store_true", help="Shuffle choices randomly")
parser.add_argument("-bs", "--batch_size", type=int, default=128, help="Number of problems to process in each batch. Decrease to prevent potential OOM errors")
model_init.add_args(parser)
args = parser.parse_args()

# Model and Cache Initialization
def initialize_model_and_cache(args):
    try:
        model_init.check_args(args)
        model_init.print_options(args)
        model, tokenizer = model_init.init(args, allow_auto_split=True, progress=True, max_output_len=1, max_input_len=2048)
    except Exception as e:
        print(f"Error initializing model: {e}")
        sys.exit(1)

    cache_type = {
        "q4": ExLlamaV2Cache_Q4,
        "q6": ExLlamaV2Cache_Q6,
        "q8": ExLlamaV2Cache_Q8
    }.get(next((k[6:] for k, v in vars(args).items() if k.startswith('cache_q') and v), None), ExLlamaV2Cache)

    cache = cache_type(model, lazy=not model.loaded, max_seq_len=args.cache_size or model.config.max_seq_len)
    if not model.loaded:
        model.load_autosplit(cache, progress=True)

    return model, tokenizer, cache

# Dataset Loading and Preparation
def load_and_prepare_datasets(args):
    try:
        dataset_dev = sorted(util.get_dataset("cais/mmlu", "all", "dev"), key=lambda q: q["subject"])
        dataset_all = sorted(util.get_dataset("cais/mmlu", "all", "test"), key=lambda q: q["subject"])
    except Exception as e:
        print(f"Error loading datasets: {e}")
        sys.exit(1)

    all_subjects = set(q["subject"] for q in dataset_dev)
    if args.subjects != "all":
        sel_subjects = set(args.subjects.split(","))
        invalid_subjects = sel_subjects - all_subjects
        if invalid_subjects:
            print(f"Subjects not present in dataset: {', '.join(invalid_subjects)}")
            sys.exit(1)
        all_subjects = sel_subjects

    if args.shuffle:
        for problem in dataset_all:
            if problem["subject"] in all_subjects:
                perm = random.sample(range(4), k=4)
                problem["choices"] = [problem["choices"][i] for i in perm]
                problem["answer"] = perm.index(problem["answer"])

    return dataset_dev, dataset_all, all_subjects

# Question Formatting
def format_question(question: str, choices: list[str], answer: int | None):
    f = question + "\n" + "\n".join(f"{c}. {choices[i]}" for i, c in enumerate("ABCD")) + "\nAnswer:"
    if answer is not None: f += f" {'ABCD'[answer]}\n\n"
    return f

# Preprompt Preparation
def prepare_preprompts(all_subjects, dataset_dev, tokenizer, args, progress, task_id):
    preprompt_ids = {}
    for subject in all_subjects:
        preprompt = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
        fewshots = 0
        for pq in dataset_dev:
            if fewshots == args.fewshot_examples: break
            if pq["subject"] != subject: continue
            preprompt += format_question(pq["question"], pq["choices"], pq["answer"])
        preprompt_ids[subject] = tokenizer.encode(preprompt, add_bos=True)
        progress.update(task_id, advance=1)
    return preprompt_ids

# Job Preparation
def prepare_jobs(dataset_all, all_subjects, preprompt_ids, tokenizer, gen_settings, batch_size, progress, task_id):
    jobs = []
    for q in dataset_all:
        if q["subject"] not in all_subjects: continue
        prompt = format_question(q["question"], q["choices"], None)
        prompt_ids = tokenizer.encode(prompt, add_bos=False)
        job = ExLlamaV2DynamicJob(
            input_ids=torch.cat([preprompt_ids[q["subject"]], prompt_ids], dim=-1),
            gen_settings=gen_settings,
            max_new_tokens=1,
            return_top_tokens=4,
            identifier=q,
        )
        jobs.append(job)
        progress.update(task_id, advance=1)

        if len(jobs) == batch_size:
            yield jobs
            jobs = []

    if jobs:
        yield jobs

# Batch Processing
def process_batch(generator, job_batch, token_map, token_rmap, progress, task_id):
    for job in job_batch:
        generator.enqueue(job)

    while generator.num_remaining_jobs():
        results = generator.iterate()
        for result in results:
            if not result["eos"]: continue
            top_tokens, top_probs, q = result["top_k_tokens"], result["top_k_probs"], result["identifier"]
            correct_answer = q["answer"]
            for i in range(top_tokens.shape[-1]):
                if top_tokens[0, 0, i].item() == token_map[correct_answer]:
                    confidence = top_probs[0, 0, i].item()
            q["correct_answer_confidence"] = confidence
            q["answer_correct"] = token_rmap[top_tokens[0, 0, 0].item()] == correct_answer
            progress.update(task_id, advance=1)

# Result Summarization
def summarize_results(dataset_all):
    results = defaultdict(lambda: {"total": 0, "correct": 0, "confidence_sum": 0})

    for q in dataset_all:
        if "answer_correct" not in q:
            continue
        subject = q["subject"]
        results[subject]["total"] += 1
        results["overall"]["total"] += 1
        if q["answer_correct"]:
            results[subject]["correct"] += 1
            results["overall"]["correct"] += 1
        results[subject]["confidence_sum"] += q["correct_answer_confidence"]
        results["overall"]["confidence_sum"] += q["correct_answer_confidence"]

    print("\nResults:")
    print(f"{'Subject':<30} {'Accuracy':<10} {'Confidence':<10}")
    print("-" * 50)
    for subject, data in sorted(results.items()):
        if subject != "overall":
            acc = data["correct"] / data["total"] * 100
            conf = data["confidence_sum"] / data["total"] * 100
            print(f"{subject:<30} {acc:.2f}%      {conf:.2f}%")

    print("-" * 50)
    overall = results["overall"]
    overall_acc = overall["correct"] / overall["total"] * 100
    overall_conf = overall["confidence_sum"] / overall["total"] * 100
    print(f"{'Overall':<30} {overall_acc:.2f}%      {overall_conf:.2f}%")

# Main Execution
model, tokenizer, cache = initialize_model_and_cache(args)
dataset_dev, dataset_all, all_subjects = load_and_prepare_datasets(args)

generator = ExLlamaV2DynamicGenerator(model=model, cache=cache, tokenizer=tokenizer, max_batch_size=1024, max_q_size=1)
gen_settings = ExLlamaV2Sampler.Settings(token_repetition_penalty=1.0, temperature=1.0, top_k=10, top_p=1.0)
token_map = [tokenizer.single_id(" " + c) for c in "ABCD"]
token_rmap = {token_map[i]: i for i in range(len("ABCD"))}
gen_settings.allow_tokens(tokenizer, token_map)

progress = Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(), TextColumn("{task.completed}/{task.total}"))

with progress:
    preprompt_task = progress.add_task("[red]Preparing preprompts", total=len(all_subjects))
    preprompt_ids = prepare_preprompts(all_subjects, dataset_dev, tokenizer, args, progress, preprompt_task)

    total_jobs = sum(1 for q in dataset_all if q["subject"] in all_subjects)
    preparation_task = progress.add_task("[green]Preparing questions", total=total_jobs)
    processing_task = progress.add_task("[blue]Processing questions", total=total_jobs)

    for job_batch in prepare_jobs(dataset_all, all_subjects, preprompt_ids, tokenizer, gen_settings, args.batch_size, progress, preparation_task):
        process_batch(generator, job_batch, token_map, token_rmap, progress, processing_task)

summarize_results(dataset_all)