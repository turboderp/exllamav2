
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
    model_init,
)

import argparse
import torch

from exllamav2.generator import (
    ExLlamaV2StreamingGenerator,
    ExLlamaV2Sampler
)

# Options

parser = argparse.ArgumentParser(description = "Simple Llama2 chat example for ExLlamaV2")
parser.add_argument("-mode", "--mode", choices = ["llama", "raw"], help = "Chat mode. Use llama for Llama 1/2 chat finetunes.")
parser.add_argument("-un", "--username", type = str, default = "User", help = "Username when using raw chat mode")
parser.add_argument("-bn", "--botname", type = str, default = "Chatbort", help = "Bot name when using raw chat mode")
parser.add_argument("-sp", "--system_prompt", type = str, help = "Use custom system prompt")

# Initialize model and tokenizer

model_init.add_args(parser)
args = parser.parse_args()
model_init.check_args(args)
model_init.print_options(args)
model, tokenizer = model_init.init(args)

# Create cache

cache = ExLlamaV2Cache(model)

# Prompt templates

username = args.username
botname = args.botname
system_prompt = args.system_prompt
mode = args.mode

if mode == "llama":

    if not system_prompt:

        system_prompt = \
        """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  """ + \
        """Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. """ + \
        """Please ensure that your responses are socially unbiased and positive in nature."""

    first_prompt = \
    """[INST] <<SYS>>\n<|system_prompt|>\n<</SYS>>\n\n<|user_prompt|> [/INST]"""

    subs_prompt = \
    """[INST] <|user_prompt|> [/INST]"""

elif mode == "raw":

    if not system_prompt:

        system_prompt = \
        f"""This is a conversation between a helpful AI assistant named {botname} and a """ + ("""user named {username}.""" if username != "User" else """user.""")

    first_prompt = \
    f"""<|system_prompt|>\n{username}: <|user_prompt|>\n{botname}:"""

    subs_prompt = \
    f"""{username}: <|user_prompt|>\n{botname}:"""

else:

    print(" ## Error: Incorrect/no mode specified.")
    sys.exit()

# Chat context

def format_prompt(user_prompt, first):
    global system_prompt, first_prompt, subs_prompt

    if first:
        return first_prompt \
            .replace("<|system_prompt|>", system_prompt) \
            .replace("<|user_prompt|>", user_prompt)
    else:
        return subs_prompt \
            .replace("<|user_prompt|>", user_prompt)

def encode_prompt(text):
    global tokenizer, mode

    if mode == "llama":
        return tokenizer.encode(text, add_bos = True)

    if mode == "raw":
        return tokenizer.encode(text)

user_prompts = []
responses_ids = []

def get_tokenized_context(max_len):
    global user_prompts, responses_ids

    while True:

        context = torch.empty((1, 0), dtype=torch.long)

        for turn in range(len(user_prompts)):

            up_ids = encode_prompt(format_prompt(user_prompts[turn], context.shape[-1] == 0))
            context = torch.cat([context, up_ids], dim=-1)

            if turn < len(responses_ids):
                context = torch.cat([context, responses_ids[turn]], dim=-1)

        if context.shape[-1] < max_len: return context

        # If the context is too long, remove the first Q/A pair and try again. The system prompt will be moved to
        # the first entry in the truncated context

        user_prompts = user_prompts[1:]
        responses_ids = responses_ids[1:]


# Generator

generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)

settings = ExLlamaV2Sampler.Settings()
settings.temperature = 0.85
settings.top_k = 50
settings.top_p = 0.8
settings.token_repetition_penalty = 1.05

max_response_tokens = 1024
min_space_in_context = 256

# Stop conditions

if mode == "llama":

    generator.set_stop_conditions([tokenizer.eos_token_id])

if mode == "raw":

    generator.set_stop_conditions([username + ":", username[0:1] + ":"])

# Color codes

col_default = "\u001b[0m"
col_user = "\u001b[33;1m"  # Yellow
col_bot = "\u001b[34;1m"  # Blue
col_error = "\u001b[31;1m"  # Magenta

# Main loop

while True:

    # Get user prompt

    print()
    up = input(col_user + username + ": " + col_default).strip()
    print()

    # Add to context

    user_prompts.append(up)

    # Send tokenized context to generator

    active_context = get_tokenized_context(model.config.max_seq_len - min_space_in_context)
    generator.begin_stream(active_context, settings)

    # print("------")
    # print(tokenizer.decode(active_context))
    # print("------")

    # Stream response

    if mode == "raw":

        print(col_bot + botname + ": " + col_default, end = "")

    response_tokens = 0
    response_text = ""
    responses_ids.append(torch.empty((1, 0), dtype = torch.long))

    while True:

        # Get response stream

        chunk, eos, tokens = generator.stream()
        if len(response_text) == 0: chunk = chunk.lstrip()
        response_text += chunk
        responses_ids[-1] = torch.cat([responses_ids[-1], tokens], dim = -1)
        print(chunk, end="")
        sys.stdout.flush()

        # If model has run out of space, rebuild the context and restart stream

        if generator.full():

            active_context = get_tokenized_context(model.config.max_seq_len - min_space_in_context)
            generator.begin_stream(active_context, settings)

        # If response is too long, cut it short and append EOS

        response_tokens += 1
        if response_tokens == max_response_tokens:

            responses_ids[-1] = torch.cat([responses_ids[-1], tokenizer.single_token(tokenizer.eos_token_id)], dim = -1)

            print()
            print(col_error + f" !! Response exceeded {max_response_tokens} tokens and was cut short." + col_default)
            break

        # EOS signal returned

        if eos:

            if mode == "llama":
                print()
            #     responses_ids[-1] = torch.cat([responses_ids[-1], tokenizer.single_token(tokenizer.eos_token_id)], dim = -1)

            break

