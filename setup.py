from setuptools import setup, Extension
from torch.utils import cpp_extension
from torch import version as torch_version
import os

extension_name = "exllamav2_ext"
verbose = False
ext_debug = False

precompile = 'EXLLAMA_NOCOMPILE' not in os.environ

windows = (os.name == "nt")

extra_cflags = ["/Ox"] if windows else ["-O3"]

if ext_debug:
    extra_cflags += ["-ftime-report", "-DTORCH_USE_CUDA_DSA"]

extra_cuda_cflags = ["-lineinfo", "-O3"]

if torch_version.hip:
    extra_cuda_cflags += ["-DHIPBLAS_USE_HIP_HALF"]

extra_compile_args = {
    "cxx": extra_cflags,
    "nvcc": extra_cuda_cflags,
}

setup_kwargs = {
    "ext_modules": [
        cpp_extension.CUDAExtension(
            extension_name,
            [
                "exllamav2/exllamav2_ext/ext.cpp",
                "exllamav2/exllamav2_ext/cuda/h_gemm.cu",
                "exllamav2/exllamav2_ext/cuda/lora.cu",
                "exllamav2/exllamav2_ext/cuda/pack_tensor.cu",
                "exllamav2/exllamav2_ext/cuda/quantize.cu",
                "exllamav2/exllamav2_ext/cuda/q_matrix.cu",
                "exllamav2/exllamav2_ext/cuda/q_attn.cu",
                "exllamav2/exllamav2_ext/cuda/q_mlp.cu",
                "exllamav2/exllamav2_ext/cuda/q_gemm.cu",
                "exllamav2/exllamav2_ext/cuda/rms_norm.cu",
                "exllamav2/exllamav2_ext/cuda/rope.cu",
                "exllamav2/exllamav2_ext/cuda/cache.cu",
                "exllamav2/exllamav2_ext/cuda/util.cu",
                "exllamav2/exllamav2_ext/cuda/comp_units/kernel_select.cu",
                "exllamav2/exllamav2_ext/cuda/comp_units/unit_gptq_1.cu",
                "exllamav2/exllamav2_ext/cuda/comp_units/unit_gptq_2.cu",
                "exllamav2/exllamav2_ext/cuda/comp_units/unit_gptq_3.cu",
                "exllamav2/exllamav2_ext/cuda/comp_units/unit_exl2_1.cu",
                "exllamav2/exllamav2_ext/cuda/comp_units/unit_exl2_2.cu",
                "exllamav2/exllamav2_ext/cuda/comp_units/unit_exl2_3.cu",
                "exllamav2/exllamav2_ext/cpp/quantize_func.cpp",
                "exllamav2/exllamav2_ext/cpp/sampling.cpp"
            ],
            extra_compile_args=extra_compile_args,
            libraries=["cublas"] if windows else [],
        )],
    "cmdclass": {"build_ext": cpp_extension.BuildExtension}
} if precompile else {}

version_py = {}
with open("exllamav2/version.py", encoding = "utf8") as fp:
    exec(fp.read(), version_py)
version = version_py["__version__"]
print("Version:", version)

# version = "0.0.5"

setup(
    name = "exllamav2",
    version = version,
    packages = [
        "exllamav2",
        "exllamav2.generator",
        # "exllamav2.generator.filters",
        # "exllamav2.server",
        # "exllamav2.exllamav2_ext",
        # "exllamav2.exllamav2_ext.cpp",
        # "exllamav2.exllamav2_ext.cuda",
        # "exllamav2.exllamav2_ext.cuda.quant",
    ],
    url = "https://github.com/turboderp/exllamav2",
    license = "MIT",
    author = "turboderp",
    install_requires = [
        "pandas",
        "ninja",
        "fastparquet",
        "torch>=2.0.1",
        "safetensors>=0.3.2",
        "sentencepiece>=0.1.97",
        "pygments",
        "websockets",
        "regex"
    ],
    include_package_data = True,
    verbose = verbose,
    **setup_kwargs,
)
