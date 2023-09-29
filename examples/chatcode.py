import os
import re
import sys
from io import StringIO

from pygments import highlight
from pygments.formatter import Formatter
from pygments.formatters.terminal import TerminalFormatter
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.style import Style
from pygments.styles.default import DefaultStyle
from pygments.token import Token
from pygments.util import ClassNotFound

import shutil

# Append the parent directory to sys.path
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
parser.add_argument("-mode", "--mode", choices = ["llama", "raw", "codellama"], help = "Chat mode. Use llama for Llama 1/2 chat finetunes.")
parser.add_argument("-un", "--username", type = str, default = "User", help = "Username when using raw chat mode")
parser.add_argument("-bn", "--botname", type = str, default = "Chatbort", help = "Bot name when using raw chat mode")
parser.add_argument("-sp", "--system_prompt", type = str, help = "Use custom system prompt")

parser.add_argument("-temp", "--temperature", type = float, default = 0.95, help = "Sampler temperature, default = 0.95 (1 to disable)")
parser.add_argument("-topk", "--top_k", type = int, default = 50, help = "Sampler top-K, default = 50 (0 to disable)")
parser.add_argument("-topp", "--top_p", type = float, default = 0.8, help = "Sampler top-P, default = 0.8 (0 to disable)")
parser.add_argument("-typical", "--typical", type = float, default = 0.0, help = "Sampler typical threshold, default = 0.0 (0 to disable)")
parser.add_argument("-repp", "--repetition_penalty", type = float, default = 1.1, help = "Sampler repetition penalty, default = 1.1 (1 to disable)")
parser.add_argument("-maxr", "--max_response_tokens", type = int, default = 1000, help = "Max tokens per response, default = 1000")
parser.add_argument("-resc", "--response_chunk", type = int, default = 250, help = "Space to reserve in context for reply, default = 250")

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

if mode == "llama" or mode == "codellama":

    if not system_prompt:

        if mode == "llama":

            system_prompt = \
            """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  """ + \
            """Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. """ + \
            """Please ensure that your responses are socially unbiased and positive in nature."""

        elif mode == "codellama":

            system_prompt = \
            """You are a helpful coding assistant. Always answer as helpfully as possible."""
            
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

    if mode == "llama" or mode == "codellama":
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
settings.temperature = args.temperature
settings.top_k = args.top_k
settings.top_p = args.top_p
settings.typical = args.typical
settings.token_repetition_penalty = args.repetition_penalty

max_response_tokens = args.max_response_tokens
min_space_in_context = args.response_chunk

# Stop conditions

if mode == "llama" or mode == "codellama":

    generator.set_stop_conditions([tokenizer.eos_token_id])

if mode == "raw":

    generator.set_stop_conditions([username + ":", username[0:1] + ":", username.upper() + ":", username.lower() + ":", tokenizer.eos_token_id])

# ANSI color codes

col_default = "\u001b[0m"
col_user = "\u001b[34;1m"  # Blue
col_bot = "\u001b[31;1m"  # Bright Red
col_error = "\u001b[31;1m"  # Magenta

# Code block syntax helpers

in_code_block = False
code_block_text = ""
lines_printed = 0

code_pad = 2
block_pad_left = 1

# Code block formatter for black background

class BlackBackgroundTerminalFormatter(TerminalFormatter):
    def format(self, tokensource, outfile):
        global code_pad, block_pad_left
        # Create a buffer to capture the parent class's output
        buffer = StringIO()
        # Call the parent class's format method
        super().format(tokensource, buffer)
        # Get the content from the buffer
        content = buffer.getvalue()

        # Padding of code
        lines = content.split('\n')
        padded_lines = [f"{lines[0]}{' '*code_pad*2}"] + [f"{' '*code_pad}{line}{' '*code_pad}" for line in lines[1:-1]] + [lines[-1]]
        content = '\n'.join(padded_lines)

        # Modify the ANSI codes to include a black background
        modified_content = self.add_black_background(content)

        # Offset codeblock
        modified_content = '\n'.join([modified_content.split('\n')[0]] + [f"{' '*block_pad_left}{line}" for line in modified_content.split('\n')[1:]])

        # Relay the modified content to the outfile
        outfile.write(modified_content)

    def add_black_background(self, content):
        # Split the content into lines
        lines = content.split('\n')

        # Process each line to ensure it has a black background
        processed_lines = []
        for line in lines:
            # Split the line into tokens based on ANSI escape sequences
            tokens = re.split(r'(\033\[[^m]*m)', line)
            # Process each token to ensure it has a black background
            processed_tokens = []
            for token in tokens:
                # If the token is an ANSI escape sequence
                if re.match(r'\033\[[^m]*m', token):
                    # Append the black background code to the existing ANSI code
                    processed_tokens.append(f'{token}\033[40m')
                else:
                    # If the token is not an ANSI escape sequence, add the black background code to it
                    processed_tokens.append(f'\033[40m{token}\033[0m')  # Reset code added here

            # Join the processed tokens back into a single line
            processed_line = ''.join(processed_tokens)
            # Add the ANSI reset code to the end of the line
            processed_line += '\033[0m'
            processed_lines.append(processed_line)

        # Join the processed lines back into a single string
        modified_content = '\n'.join(processed_lines)

        return modified_content

# Print a code block, updating the CLI in real-time

def print_code_block(chunk):
    global lines_printed
    global code_block_text
    global code_pad, block_pad_left

    # Clear previously printed lines
    for _ in range(lines_printed):  # -1 not needed?
        # Move cursor up one line
        print('\x1b[1A', end='')
        # Clear line
        print('\x1b[2K', end='')

    terminal_width = shutil.get_terminal_size().columns

    # Check if the chunk will exceed the terminal width on the current line
    current_line_length = len(code_block_text.split('\n')[-1]) + len(chunk) + 2 * 3 + 3  # Including padding and offset
    if current_line_length > terminal_width:
        code_block_text += '\n'

    # Update the code block text
    code_block_text += chunk

    # Remove language after codeblock start
    code_block_text = re.sub(r'```.*?$', '```', code_block_text, flags=re.MULTILINE)

    # Split updated text into lines and find the longest line
    lines = code_block_text.split('\n')
    max_length = max(len(line) for line in lines)

    # Pad all lines to match the length of the longest line
    padded_lines = [line.ljust(max_length) for line in lines]
    
    # Join padded lines into a single string
    padded_text = '\n'.join(padded_lines)

    # Try guessing the lexer for syntax highlighting
    try:
        lexer = guess_lexer(padded_text)
    except ClassNotFound:
        lexer = get_lexer_by_name("text")  # Fallback to plain text if language isn't supported by pygments

    formatter = BlackBackgroundTerminalFormatter()
    highlighted_text = highlight(padded_text, lexer, formatter)

    highlighted_text = highlighted_text.replace('\n', '\033[0m\n')


    # Print the updated padded and highlighted text
    print(highlighted_text, end='')
    
    # Update the lines_printed counter
    lines_printed = len(lines)

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
        
        # Check for code block delimiters
        if chunk.startswith("```"):
            in_code_block = not in_code_block  # Toggle in_code_block flag
            chunk = chunk[3:]  # Remove the delimiter from the chunk
            print('\n')
        
        if in_code_block:
            print_code_block(chunk)  # Handle code block streaming
        else:
            # If exiting a code block, highlight and print the code block text
            if code_block_text:
                code_block_text = ""  # Reset code_block_text for the next code block
                lines_printed = 0
                print('\033[0m', end='')    # Reset block color to be certain

            # Continue as normal if not in a code block
            responses_ids[-1] = torch.cat([responses_ids[-1], tokens], dim=-1)
            print(chunk, end="")
            sys.stdout.flush()

        # If model has run out of space, rebuild the context and restart stream

        if generator.full():

            active_context = get_tokenized_context(model.config.max_seq_len - min_space_in_context)
            generator.begin_stream(active_context, settings)

        # If response is too long, cut it short, and append EOS if that was a stop condition

        response_tokens += 1
        if response_tokens == max_response_tokens:

            if tokenizer.eos_token_id in generator.stop_tokens:
                responses_ids[-1] = torch.cat([responses_ids[-1], tokenizer.single_token(tokenizer.eos_token_id)], dim = -1)

            print()
            print(col_error + f" !! Response exceeded {max_response_tokens} tokens and was cut short." + col_default)
            break

        # EOS signal returned

        if eos:

            if mode == "llama" or mode == "codellama":
                print()

            break

