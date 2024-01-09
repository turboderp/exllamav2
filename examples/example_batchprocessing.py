import sys
import os
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
)

from exllamav2.generator import (
    ExLlamaV2BaseGenerator,
    ExLlamaV2Sampler,
)

import time
import json

# Initialize model and cache

model_directory = "airoboros-l2-70b-3.1-4.0bpw-h6-exl2"

config = ExLlamaV2Config()
config.model_dir = model_directory
config.prepare()

model = ExLlamaV2(config)
print("Loading model: " + model_directory)

# allocate 18 GB to CUDA:0 and 24 GB to CUDA:1.
# (Call `model.load()` if using a single GPU.)

tokenizer = ExLlamaV2Tokenizer(config)


model.load([18, 22])
cache = ExLlamaV2Cache(model, batch_size=4)
# Initialize generator

# Generate some text

settings = ExLlamaV2Sampler.Settings()
settings.temperature = 0.6
settings.top_k = 0
settings.top_p = 0.9
settings.token_repetition_penalty = 1.1
settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])

max_new_tokens = 150
generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)

generator.warmup()
tmpCollector = {}
# Open JSON file
with open('sample4Sum.json') as tmpinputCollector:
    inputCollector = json.load(tmpinputCollector)


with open('sample4Sum.json', 'r+') as tmpinputCollector:
    for index, row in tqdm(enumerate(inputCollector), total=len(inputCollector), desc="Processing"):
        output = ''
        
        if row['system'] != 'fill':
            continue
        #print('working on', row)
        content = ' '.join(row['txt'].split(' ')[:2000])
        prompt = f'Below describes a chat between an user and an assistant:\nUser: I will issue command and you will respond.\nAssistant: Understood.\nUser: Summarize the text in about 140 words, focusing on events and actions. And infer the appearance and personality of the characters involved in a few sentences if possible. Write confidently, objectively and professionally, in one paragraph, and end with "EndofSummary". Here goes the quoted text:"\n```\n{content}\n```\nAgain, remember your task is to summarize the quoted text provided just above in about 140 words, focusing on events and actions. And infer the appearance and personality of the characters involved in a few sentences if possible, write confidently objectively and professionally, in one paragraph, and end with "EndofSummary".\nAssistant: Understood. Including any character description if available, I will summarize your quoted text in one paragraph while keeping my tone professional, and lastly, end with "EndofSummary". The quoted text is mainly talking about'
        tmpCollector[index] = prompt
        if len(tmpCollector) < 4:
            continue
        output = generator.generate_simple(list(tmpCollector.values()), settings, 500)
        for singleComp in output:
            for singlePrompt in tmpCollector:
                initialPrompt = tmpCollector[singlePrompt]
                if not initialPrompt in singleComp:
                    continue
                if 'EndofSummary' not in singleComp:
                    break
                completion = singleComp.split('"EndofSummary".\nAssistant: Understood. Including any character description if available, I will summarize your quoted text in one paragraph while keeping my tone professional, and lastly, end with "EndofSummary". The quoted text is mainly talking about')[1].split('EndofSummary')[0].split('\n')[0]
                if not completion.endswith('.'):
                    completion = completion.split('.')[:-1]
                    completion = '.'.join(completion)
                if len(completion.split(' ')) <=30:
                    break
                if 'Please' in completion:
                    break
                if 'please' in completion:
                    break
                print('>>>>>>>>>>>>'+completion)
                print()
                inputCollector[index]={'system': 'This text below is about' + completion, 'txt': inputCollector[index]['txt']}
                break
        tmpCollector = {}
        tmpinputCollector.seek(0)
        json.dump(inputCollector, tmpinputCollector)
        tmpinputCollector.truncate()
# Save the result to a JSON file
with open("sample4Sum.json", "w") as outfile:
    outfile.seek(0)
    json.dump(inputCollector, outfile)
