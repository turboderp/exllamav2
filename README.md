# ExLlamaV2

ExLlamaV2 is an inference library for running local LLMs on modern consumer GPUs.


## Overview of differences compared to V1

- Faster, better kernels
- Cleaner and more versatile codebase
- Support for a new quant format (see below)


## Performance

Some quick tests to compare performance with V1. There may be more performance optimizations in the future, and
speeds will vary across GPUs, with slow CPUs still being a potential bottleneck:

| Model      | Mode         | Size  | grpsz | act | V1: 3090Ti | V1: 4090 | V2: 3090Ti | V2: 4090    |
|------------|--------------|-------|-------|-----|------------|----------|------------|-------------|
| Llama      | GPTQ         | 7B    | 128   | no  | 143 t/s    | 173 t/s  | 175 t/s    | **195** t/s |
| Llama      | GPTQ         | 13B   | 128   | no  | 84 t/s     | 102 t/s  | 105 t/s    | **110** t/s |
| Llama      | GPTQ         | 33B   | 128   | yes | 37 t/s     | 45 t/s   | 45 t/s     | **48** t/s  |
| OpenLlama  | GPTQ         | 3B    | 128   | yes | 194 t/s    | 226 t/s  | 295 t/s    | **321** t/s |
| CodeLlama  | EXL2 4.0 bpw | 34B   | -     | -   | -          | -        | 42 t/s     | **48** t/s  |
| Llama2     | EXL2 3.0 bpw | 7B    | -     | -   | -          | -        | 195 t/s    | **224** t/s |
| Llama2     | EXL2 4.0 bpw | 7B    | -     | -   | -          | -        | 164 t/s    | **197** t/s |
| Llama2     | EXL2 5.0 bpw | 7B    | -     | -   | -          | -        | 144 t/s    | **160** t/s |
| Llama2     | EXL2 2.5 bpw | 70B   | -     | -   | -          | -        | 30 t/s     | **35** t/s  |
| TinyLlama  | EXL2 3.0 bpw | 1.1B  | -     | -   | -          | -        | 536 t/s    | **635** t/s |
| TinyLlama  | EXL2 4.0 bpw | 1.1B  | -     | -   | -          | -        | 509 t/s    | **590** t/s |


## How to

Clone the repository and install dependencies:

```
git clone https://github.com/turboderp/exllamav2
cd exllamav2
pip install -r requirements.txt

python test_inference.py -m <path_to_model> -p "Once upon a time,"
```

A simple console chatbot is included. Run it with:

```
python examples/chat.py -m <path_to_model> -mode llama
```


The `-mode` argument chooses the prompt format to use. `llama` is for the Llama(2)-chat finetunes, while `codellama`
probably works better for CodeLlama-instruct. `raw` will produce a simple chatlog-style chat that works with base 
models and various other finetunes. You can also provide a custom system prompt with `-sp`. 


## Integration and APIs

- [TabbyAPI](https://github.com/theroyallab/tabbyAPI/) is a FastAPI-based server that provides an OpenAI-style web API
compatible with [SillyTavern](https://sillytavernai.com/) and other frontends.  

- [ExUI](https://github.com/turboderp/exui) is a simple, standalone single-user web UI that serves an ExLlamaV2 instance
directly with chat and notebook modes.

- [text-generation-webui](https://github.com/oobabooga/text-generation-webui) supports ExLlamaV2 through the **exllamav2**
and **exllamav2_HF** loaders.


## Installation

### Method 1: Install from source

To install the current dev version, clone the repo and run the setup script:

```
git clone https://github.com/turboderp/exllamav2
cd exllamav2
python setup.py install --user
```

By default this will also compile and install the Torch C++ extension (`exllamav2_ext`) that the library relies on. 
You can skip this step by setting the `EXLLAMA_NOCOMPILE` environment variable:

```
EXLLAMA_NOCOMPILE= python setup.py install --user
```

This will install the "JIT version" of the package, i.e. it will install the Python components without building the
C++ extension in the process. Instead, the extension will be built the first time the library is used, then cached in 
`~/.cache/torch_extensions` for subsequent use.

### Method 2: Install from release (with prebuilt extension)

Releases are available [here](https://github.com/turboderp/exllamav2/releases), with prebuilt wheels that contain the
extension binaries. Make sure to grab the right version, matching your platform, Python version (`cp`) and CUDA version.
Download an appropriate wheel, then run:

```
pip install exllamav2-0.0.4+cu118-cp310-cp310-linux_x86_64.whl
```

The `py3-none-any.whl` version is the JIT version which will build the extension on first launch. The `.tar.gz` file
can also be installed this way, and it will build the extension while installing.

### Method 3: Install from PyPI

A PyPI package is available as well. It can be installed with:

```
pip install exllamav2
```

The version available through PyPI is the JIT version (see above). Still working on a solution for distributing
prebuilt wheels via PyPI.


## EXL2 quantization

ExLlamaV2 supports the same 4-bit GPTQ models as V1, but also a new "EXL2" format. EXL2 is based on the same
optimization method as GPTQ and supports 2, 3, 4, 5, 6 and 8-bit quantization. The format allows for mixing quantization
levels within a model to achieve any average bitrate between 2 and 8 bits per weight.

Moreover, it's possible to apply multiple quantization levels to each linear layer, producing something akin to sparse 
quantization wherein more important weights (columns) are quantized with more bits. The same remapping trick that lets
ExLlama work efficiently with act-order models allows this mixing of formats to happen with little to no impact on
performance.

Parameter selection is done automatically by quantizing each matrix multiple times, measuring the quantization 
error (with respect to the chosen calibration data) for each of a number of possible settings, per layer. Finally, a
combination is chosen that minimizes the maximum quantization error over the entire model while meeting a target
average bitrate.

In my tests, this scheme allows Llama2 70B to run on a single 24 GB GPU with a 2048-token context, producing coherent 
and mostly stable output with 2.55 bits per weight. 13B models run at 2.65 bits within 8 GB of VRAM, although currently
none of them uses GQA which effectively limits the context size to 2048. In either case it's unlikely that the model
will fit alongside a desktop environment. For now.

[![chat_screenshot](doc/llama2_70b_chat_thumb.png)](doc/llama2_70b_chat.png)
[![chat_screenshot](doc/codellama_13b_instruct_thumb.png)](doc/codellama_13b_instruct.png)

### Conversion

A script is provided to quantize models. Converting large models can be somewhat slow, so be warned. The conversion
script and its options are explained in [detail here](doc/convert.md)

### HuggingFace repos

- I've uploaded a few EXL2-quantized models to Hugging Face to play around with, [here](https://huggingface.co/turboderp).

- [LoneStriker](https://huggingface.co/LoneStriker) provides a large number of EXL2 models on Hugging Face. 
