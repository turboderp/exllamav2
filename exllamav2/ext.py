import torch
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.utils.cpp_extension import load
import os
import sys
import platform

library_dir = os.path.dirname(os.path.abspath(__file__))
extension_name = "exllamav2_ext"
verbose = False

# Another kludge to get things compiling in Windows

windows = os.name == "nt"
if windows:
    def find_msvc():
        for msvc_dir in [a + "\\Microsoft Visual Studio\\" + b + "\\" + c + "\\VC\Tools\\MSVC\\"
                         for b in ["2022", "2019", "2017"]
                         for a in [os.environ["ProgramW6432"], os.environ["ProgramFiles(x86)"]]
                         for c in ["BuildTools", "Community", "Professional", "Enterprise", "Preview"]
                         ]:
            if not os.path.exists(msvc_dir):
                continue
            versions = sorted(os.listdir(msvc_dir), reverse = True)
            for version in versions:
                compiler_dir = msvc_dir + version + "\\bin\\Hostx64\\x64"
                if os.path.exists(compiler_dir) and os.path.exists(compiler_dir + "\\cl.exe"):
                    return compiler_dir
        return None

    import subprocess

    try:
        subprocess.check_output(["where", "/Q", "cl"])
    except subprocess.CalledProcessError as e:
        cl_path = find_msvc()
        if cl_path:
            if verbose:
                print("Injected compiler path:", cl_path)
            os.environ["path"] += ";" + cl_path
        else:
            print("Unable to find cl.exe; compilation will probably fail.", file = sys.stderr)

exllamav2_ext = load \
(
    name = extension_name,
    sources = [
        os.path.join(library_dir, "exllamav2_ext/ext.cpp"),
        os.path.join(library_dir, "exllamav2_ext/cuda/pack_tensor.cu"),
        os.path.join(library_dir, "exllamav2_ext/cuda/adjust_scale.cu"),
        os.path.join(library_dir, "exllamav2_ext/cuda/quantize.cu"),
        os.path.join(library_dir, "exllamav2_ext/cuda/q_matrix.cu"),
        os.path.join(library_dir, "exllamav2_ext/cuda/q_attn.cu"),
        os.path.join(library_dir, "exllamav2_ext/cuda/q_mlp.cu"),
        os.path.join(library_dir, "exllamav2_ext/cuda/q_gemm.cu"),
        os.path.join(library_dir, "exllamav2_ext/cuda/rms_norm.cu"),
        os.path.join(library_dir, "exllamav2_ext/cuda/rope.cu"),
        os.path.join(library_dir, "exllamav2_ext/cpp/quantize_func.cpp"),
        os.path.join(library_dir, "exllamav2_ext/cpp/torch_attn.cpp"),
        os.path.join(library_dir, "exllamav2_ext/cpp/sampling.cpp"),
        # ..
    ],
    extra_include_paths = [os.path.join(library_dir, "exllamav2_ext")],
    verbose = verbose,
    extra_ldflags=(["cublas.lib"] + ([f"/LIBPATH:{os.path.join(sys.base_prefix, 'libs')}"] if sys.base_prefix != sys.prefix else [])) if windows else [],
    extra_cuda_cflags=["-lineinfo", "-O3"] + (["-U__HIP_NO_HALF_CONVERSIONS__"] if torch.version.hip else []),

        # "-maxrregcount=128"

    extra_cflags=["/Ox" if windows else "-O3"]
    # extra_cflags = ["-ftime-report", "-DTORCH_USE_CUDA_DSA"]
)

ext_c = exllamav2_ext


# Dummy tensor to pass to C++ extension in place of None/NULL

none_tensor = torch.empty((1, 1), device = "meta")


# Create Q matrix

def make_q_matrix(w: dict, temp_dq, key: str = None):

    w["q_scale_max"] /= 256
    w["q_perm"] = w["q_perm"].short()
    w["q_invperm"] = w["q_invperm"].short()

    return ext_c.make_q_matrix(w["q_weight"],
                               w["q_perm"],
                               w["q_invperm"],
                               w["q_scale"],
                               w["q_scale_max"],
                               w["q_groups"],
                               temp_dq)
