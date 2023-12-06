from llama import LlamaForCausalLM
import transformers
from transformers import LlamaTokenizer

PATH = '/media/storage/models/Llama-2-13b-chat-E8P-2Bit'
model = LlamaForCausalLM.from_pretrained(
    PATH, torch_dtype='auto', low_cpu_mem_usage=True, use_flash_attention_2=True, device_map='auto').half()
model_str = transformers.LlamaConfig.from_pretrained(PATH)._name_or_path
print('Model Loaded: ', model_str)
tokenizer = LlamaTokenizer.from_pretrained(model_str)
tokenizer.pad_token = tokenizer.eos_token

inputs = tokenizer('once upon a time,', return_tensors='pt')
outputs = model.generate(input_ids=inputs['input_ids'].cuda(),
                         attention_mask=inputs['attention_mask'].cuda(),
                         max_length=128,
                         penalty_alpha=0.6,
                         top_k=4,
                         return_dict_in_generate=True).sequences[0]
print()
print('Model Output: ', tokenizer.decode(outputs, skip_special_tokens=True))