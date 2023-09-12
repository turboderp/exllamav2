import torch
from torch.utils.cpp_extension import load
import os
import sys
import platform

extension_name = "exllamav2_ext"
verbose = False
ext_debug = False

windows = (os.name == "nt")

# Kludge to get compilation working on Windows

if windows:

    def find_msvc():

        # Possible locations for MSVC, in order of preference

        program_files_x64 = os.environ["ProgramW6432"]
        program_files_x86 = os.environ["ProgramFiles(x86)"]

        msvc_dirs = \
        [
            a + "\\Microsoft Visual Studio\\" + b + "\\" + c + "\\VC\Tools\\MSVC\\"
            for b in ["2022", "2019", "2017"]
            for a in [program_files_x64, program_files_x86]
            for c in ["BuildTools", "Community", "Professional", "Enterprise", "Preview"]
        ]

        for msvc_dir in msvc_dirs:
            if not os.path.exists(msvc_dir): continue

            # Prefer the latest version

            versions = sorted(os.listdir(msvc_dir), reverse = True)
            for version in versions:

                compiler_dir = msvc_dir + version + "\\bin\\Hostx64\\x64"
                if os.path.exists(compiler_dir) and os.path.exists(compiler_dir + "\\cl.exe"):
                    return compiler_dir

        # No path found

        return None

    import subprocess

    # Check if cl.exe is already in the path

    try:

        subprocess.check_output(["where", "/Q", "cl"])

    # If not, try to find an installation of Visual Studio and append the compiler dir to the path

    except subprocess.CalledProcessError as e:

        cl_path = find_msvc()
        if cl_path:
            if verbose:
                print(" -- Injected compiler path:", cl_path)
            os.environ["path"] += ";" + cl_path
        else:
            print(" !! Unable to find cl.exe; compilation will probably fail", file = sys.stderr)


# gcc / cl.exe flags

extra_cflags = ["/Ox"] if windows else ["-O3"]

if ext_debug:
    extra_cflags += ["-ftime-report", "-DTORCH_USE_CUDA_DSA"]


# nvcc flags

extra_cuda_cflags = ["-lineinfo", "-O3"]
# extra_cuda_cflags += ["-maxrregcount=128"]


# linker flags

extra_ldflags = []

if windows:
    extra_ldflags += ["cublas.lib"]
    if sys.base_prefix != sys.prefix:
        extra_ldflags += [f"/LIBPATH:{os.path.join(sys.base_prefix, 'libs')}"]


# sources

library_dir = os.path.dirname(os.path.abspath(__file__))
sources_dir = os.path.join(library_dir, extension_name)

sources_ = \
[
    "ext.cpp",
    "cuda/pack_tensor.cu",
    "cuda/quantize.cu",
    "cuda/q_matrix.cu",
    "cuda/q_attn.cu",
    "cuda/q_mlp.cu",
    "cuda/q_gemm.cu",
    "cuda/rms_norm.cu",
    "cuda/rope.cu",
    "cpp/quantize_func.cpp",
    "cpp/sampling.cpp"
]

sources = [os.path.join(sources_dir, s) for s in sources_]


# Load extension

exllamav2_ext = load \
(
    name = extension_name,
    sources = sources,
    extra_include_paths = [sources_dir],
    verbose = verbose,
    extra_ldflags = extra_ldflags,
    extra_cuda_cflags = extra_cuda_cflags,
    extra_cflags = extra_cflags
)

ext_c = exllamav2_ext


# Dummy tensor to pass to C++ extension in place of None/NULL

none_tensor = torch.empty((1, 1), device = "meta")


# Create Q matrix

def make_q_matrix(w: dict, temp_dq, key: str = None):

    # EXL2

    if "q_weight" in w:

        w["q_scale_max"] /= 256
        w["q_perm"] = w["q_perm"].short()
        w["q_invperm"] = w["q_invperm"].short()

        return ext_c.make_q_matrix(w["q_weight"],
                                   w["q_perm"],
                                   w["q_invperm"],
                                   w["q_scale"],
                                   w["q_scale_max"],
                                   w["q_groups"],
                                   none_tensor,
                                   none_tensor,
                                   none_tensor,
                                   temp_dq)


    # GPTQ

    elif "qweight" in w:

        if w["scales"].dtype == torch.float: w["scales"] = w["scales"].half()

        # GPTQ with g_idx (act_order)

        if "g_idx" in w and not (w["g_idx"] == 0).all().item():

            w["q_perm"] = torch.empty((w["qweight"].shape[0] * 8,), dtype = torch.short, device = w["qweight"].device)
            w["q_invperm"] = torch.empty_like(w["q_perm"])

            return ext_c.make_q_matrix(w["qweight"],
                                       w["q_perm"],
                                       w["q_invperm"],
                                       none_tensor,
                                       none_tensor,
                                       none_tensor,
                                       w["qzeros"],
                                       w["scales"],
                                       w["g_idx"].cpu(),
                                       temp_dq)

        # GPTQ without g_idx

        else:

            return ext_c.make_q_matrix(w["qweight"],
                                       none_tensor,
                                       none_tensor,
                                       none_tensor,
                                       none_tensor,
                                       none_tensor,
                                       w["qzeros"],
                                       w["scales"],
                                       none_tensor,
                                       temp_dq)


