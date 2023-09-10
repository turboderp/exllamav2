# ExLlamaV2

This is a very early release of ExLlamaV2, an inference library for running local LLMs on modern consumer GPUs. It 
still needs a lot of testing and tuning, and a few key features are not yet implemented.


## Overview of differences compared to V1

- Faster, better kernels
- Cleaner and more versatile codebase
- Support for a new quant format (see below)


## How to do stuff

Clone the repository and install dependencies:

```
git clone https://github.com/turboderp/exllamav2
cd exllamav2
pip install -r requirements.txt

python test_inference -m <path_to_model> -p "Once upon a time,"
```

For now, a simple console chatbot is included. Run it with:

`python examples/chat.py -m <path_to_model> -mode llama`

The `-mode` argument chooses the prompt format to use. `llama` is for the Llama(2)-chat finetunes, while `codellama`
probably works better for CodeLlama-instruct. `raw` will produce a simple chatlog-style chat that works with base 
models and various other finetunes. You can also provide a custom system prompt with `-p`. 


## Performance

Some quick tests to compare performance with V1. There may still be more performance optimizations in the future, and
speeds will vary across GPUs, with slow CPUs still being a potential bottleneck:

| Model      | Mode        | Size  | grpsz | act | V1: 3090 | V1: 4090 | V2: 3090 | V2: 4090    |
|------------|-------------|-------|-------|-----|----------|----------|----------|-------------|
| Llama      | GPTQ        | 7B    | 128   | no  | 143 t/s  | 173 t/s  | 175 t/s  | **195** t/s |
| Llama      | GPTQ        | 13B   | 128   | no  | 84 t/s   | 102 t/s  | 105 t/s  | **110** t/s |
| Llama      | GPTQ        | 33B   | 128   | yes | 37 t/s   | 45 t/s   | 45 t/s   | **48** t/s  |
| OpenLlama  | GPTQ        | 3B    | 128   | yes | 194 t/s  | 226 t/s  | 295 t/s  | **321** t/s |
| CodeLlama  | EXL2 4.0bpw | 34B   | -     | -   | -        | -        | 42 t/s   | **48** t/s  |
| Llama2     | EXL2 3.0bpw | 7B    | -     | -   | -        | -        | 195 t/s  | **224** t/s |
| Llama2     | EXL2 4.0bpw | 7B    | -     | -   | -        | -        | 164 t/s  | **197** t/s |
| Llama2     | EXL2 5.0bpw | 7B    | -     | -   | -        | -        | 144 t/s  | **160** t/s |
| Llama2     | EXL2 2.5bpw | 70B   | -     | -   | -        | -        | 30 t/s   | **35** t/s  |
| TinyLlama2 | EXL2 3.0bpw | 1.1B  | -     | -   | -        | -        | 536 t/s  | **635** t/s |
| TinyLlama2 | EXL2 4.0bpw | 1.1B  | -     | -   | -        | -        | 509 t/s  | **590** t/s |

### Installation

Clone the repository and run `python setup.py install --user`. (PyPi package is coming, be patient.)

ExLlamaV2 relies on a Torch C++ extension for its CUDA functions, which is compiled at runtime. This means the first
time the library is used it will take 10-20 seconds (depending on your hardware) to start, but the extension gets cached
for subsequent use.


## EXL2 quantization

ExLlamaV2 supports the same 4-bit GPTQ models as V1, but also a new format. The new format is based on the same GPTQ/OBQ
optimization approach, supporting 2, 3, 4, 5, 6 and 8-bit quantization. Most notably, by mixing them you can target any
*average* bitrate from 2 up to 8 bits. This also allows for multiple quantization settings within each linear
layer, producing something akin to sparse quantization wherein more important weights (columns) are quantized with more
bits. The same remapping trick that lets ExLlamaV1 work efficiently with act-order models also allows this mixing
of formats to happen with minimal impact on performance. 

The parameter selection is done automatically by quantizing each matrix multiple times, measuring the quantization 
error (with respect to the chosen calibration data) for each of a number of possible settings, and then finally creating
a quantization scheme for the entire model that minimizes the maximum quantization error while meeting a target average
bitrate.

In my tests, this allows Llama2 70B to run on a single 24 GB GPU at the full 4096-token context, producing at least 
coherent output with 2.5 bits per weight. It can still be unstable, so it probably still needs a little optimization.
It also only *barely* fits in 24 GB, so it most likely won't work with a desktop environment running on the same GPU.

[![chat_screenshot](doc/screenshot_chat_2.5bit_thumb.png)](doc/screenshot_chat_thumb.png)

### Conversion

A script is provided to quantize models. Converting large models can be somewhat slow, so be warned.

To use it: 

```
python convert.py \
    -i <input_HF_model> \
    -o <output_work_directory> \
    -c <calibration_data_file> \
    -b <target_bits_per_weight>
```

The output directory should be empty when you start converting. The script will dump a bunch of files there as it
works, so it can resume an interrupted job if you point it to the same output directory.

After the first pass is completed, a `measurement.json` file will be written to the output directory. This can be
supplied (with the `-m` argument) to subsequent conversion jobs to skip the first pass and save some time when quantizing
the same model to different bitrates. Once complete, the quantized tensors will be compiled into `output.safetensors`,
and this file can replace the safetensors file in the original HF model.

### HuggingFace repos

I've uploaded a few EXL2-quantized models to HuggingFace, [here](https://huggingface.co/turboderp). 

### More to come

There are still things that need to be ported over from V1, and other planned features. Among them:

- PyPi package
- ROCm support
- LoRA support
- Example chat UI
- Web server
- More samplers