import sys
import importlib
import os
from setuptools import setup, Extension
from torch.utils import cpp_extension
from torch import version as torch_version

def parse_requirements(install_requires):
    return [req.split('>=')[0].strip() for req in install_requires]

install_requires_list = [
    "pandas",
    "ninja",
    "fastparquet",
    "torch>=2.0.1",
    "safetensors>=0.3.2",
    "sentencepiece>=0.1.97",
    "pygments",
    "websockets",
    "regex"
]

required_modules = parse_requirements(install_requires_list)
missing_modules = []

for module in required_modules:
    try:
        importlib.import_module(module)
    except ImportError:
        missing_modules.append(module)

if missing_modules:
    print("You may have some missing required modules:")
    for module in missing_modules:
        print(f"- {module}")
    print("\nEnsure you have activated a local Python virtual environment and installed the dependencies.")
    print("\nTo create and activate a virtual environment, follow these steps:")
    print("$ python -m venv exllamav2_env")
    print("$ source exllamav2_env/bin/activate  # On Windows use `exllamav2_env\\Scripts\\activate`")
    print("$ pip install -r requirements.txt")
    print("$ python setup.py install")
    print("\nCheck the README.md for more information on installation.")
    sys.exit(1)
try:
    from torch.utils import cpp_extension
    from torch import version as torch_version
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if len(sys.argv) == 1:
    print("To run the installer you need to enter: python setup.py install --user")
    print("Refer to the README for further instructions.")
    sys.exit(1)

try:
    from setuptools import setup, Extension
    SETUPTOOLS_AVAILABLE = True
except ImportError:
    SETUPTOOLS_AVAILABLE = False

try:
    from torch.utils import cpp_extension
    from torch import version as torch_version
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if SETUPTOOLS_AVAILABLE and TORCH_AVAILABLE:
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
                    "exllamav2/exllamav2_ext/cuda/comp_units/unit_exl2_1a.cu",
                    "exllamav2/exllamav2_ext/cuda/comp_units/unit_exl2_1b.cu",
                    "exllamav2/exllamav2_ext/cuda/comp_units/unit_exl2_2a.cu",
                    "exllamav2/exllamav2_ext/cuda/comp_units/unit_exl2_2b.cu",
                    "exllamav2/exllamav2_ext/cuda/comp_units/unit_exl2_3a.cu",
                    "exllamav2/exllamav2_ext/cuda/comp_units/unit_exl2_3b.cu",
                    "exllamav2/exllamav2_ext/cpp/quantize_func.cpp",
                    "exllamav2/exllamav2_ext/cpp/sampling.cpp"
                ],
                extra_compile_args=extra_compile_args,
                libraries=["cublas"] if windows else [],
            )],
        "cmdclass": {"build_ext": cpp_extension.BuildExtension}
    } if precompile else {}

    version_py = {}
    with open("exllamav2/version.py", encoding="utf8") as fp:
        exec(fp.read(), version_py)
    version = version_py["__version__"]
    print("Version:", version)
    setup(
        name="exllamav2",
        version=version,
        packages=[
            "exllamav2",
            "exllamav2.generator",
            # "exllamav2.generator.filters",
            # "exllamav2.server",
            # "exllamav2.exllamav2_ext",
            # "exllamav2.exllamav2_ext.cpp",
            # "exllamav2.exllamav2_ext.cuda",
            # "exllamav2.exllamav2_ext.cuda.quant",
        ],
        url="https://github.com/turboderp/exllamav2",
        license="MIT",
        author="turboderp",
        install_requires=install_requires_list,
        include_package_data=True,
        verbose=verbose,
        **setup_kwargs,
    )
else:
    if not SETUPTOOLS_AVAILABLE:
        print("Setuptools is required but not installed.")
    if not TORCH_AVAILABLE:
        print("Torch is required but not installed.")
    sys.exit(1)
