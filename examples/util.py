
def format_prompt(prompt_format, sp, p):
    if prompt_format == "llama":
        return f"<s>[INST] <<SYS>>\n{sp}\n<</SYS>>\n\n{p} [/INST]"
    elif prompt_format == "llama3":
        return (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n\n"
            f"{sp}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{p}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

def get_stop_conditions(prompt_format, tokenizer):
    if prompt_format == "llama":
        return [tokenizer.eos_token_id]
    elif prompt_format == "llama3":
        return [tokenizer.single_id("<|eot_id|>")]

