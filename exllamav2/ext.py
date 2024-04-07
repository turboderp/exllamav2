import torch
from torch.utils.cpp_extension import load
import os
import sys
import platform

extension_name = "exllamav2_ext"
verbose = False  # Print wall of text when compiling
ext_debug = False  # Compile with debug options

# Determine if we're on Windows

windows = (os.name == "nt")

# Determine if extension is already installed or needs to be built

build_jit = False
try:
    import exllamav2_ext
except ModuleNotFoundError:
    build_jit = True

if build_jit:

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

    if torch.version.hip:
        extra_cuda_cflags += ["-DHIPBLAS_USE_HIP_HALF"]

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
        "ext_bindings.cpp",
        "ext_cache.cpp",
        "ext_gemm.cpp",
        "ext_norm.cpp",
        "ext_qattn.cpp",
        "ext_qmatrix.cpp",
        "ext_qmlp.cpp",
        "ext_quant.cpp",
        "ext_rope.cpp",
        "ext_safetensors.cpp",
        "ext_sampling.cpp",
        "cuda/h_add.cu",
        "cuda/h_gemm.cu",
        "cuda/lora.cu",
        "cuda/pack_tensor.cu",
        "cuda/quantize.cu",
        "cuda/q_matrix.cu",
        "cuda/q_attn.cu",
        "cuda/q_mlp.cu",
        "cuda/q_gemm.cu",
        "cuda/rms_norm.cu",
        "cuda/head_norm.cu",
        "cuda/layer_norm.cu",
        "cuda/rope.cu",
        "cuda/cache.cu",
        "cuda/util.cu",
        "cuda/comp_units/kernel_select.cu",
        "cuda/comp_units/unit_gptq_1.cu",
        "cuda/comp_units/unit_gptq_2.cu",
        "cuda/comp_units/unit_gptq_3.cu",
        "cuda/comp_units/unit_exl2_1a.cu",
        "cuda/comp_units/unit_exl2_1b.cu",
        "cuda/comp_units/unit_exl2_2a.cu",
        "cuda/comp_units/unit_exl2_2b.cu",
        "cuda/comp_units/unit_exl2_3a.cu",
        "cuda/comp_units/unit_exl2_3b.cu",
        "cpp/quantize_func.cpp",
        "cpp/profiling.cpp",
        "cpp/sampling.cpp",
        "cpp/sampling_avx2.cpp",
        "cpp/safetensors.cpp"
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


# Group map needed for irregular group sizes

def make_group_map(q_groups: torch.Tensor, num_qrows: int) -> torch.Tensor:

    gr = q_groups.tolist()
    group_map = []
    num_groups = len(gr) // 2
    row = 0

    for i in range(num_groups):
        bits = gr[i * 2]
        if i < num_groups - 1:
            qrows = gr[i * 2 + 3] - gr[i * 2 + 1]
        else:
            qrows = num_qrows - gr[i * 2 + 1]
        rows = qrows * 32 // bits
        for j in range(rows):
            group_map += [i]
            group_map += [rows - j]

    return torch.tensor(group_map, dtype = torch.short, device = q_groups.device)


# Create Q matrix

def make_q_matrix(w: dict,
                  temp_dq: torch.Tensor,
                  key: str = None,
                  prescale: float = 1,
                  max_dq_rows = 0):

    # EXL2

    if "q_weight" in w:

        w["q_scale_max"] *= prescale / 256
        w["q_perm"] = w["q_perm"].short()
        w["q_invperm"] = w["q_invperm"].short()

        if "q_group_map" not in w:
            w["q_group_map"] = make_group_map(w["q_groups"], w["q_weight"].shape[0])

        return ext_c.make_q_matrix(w["q_weight"],
                                   w["q_perm"],
                                   w["q_invperm"],
                                   w["q_scale"],
                                   w["q_scale_max"],
                                   w["q_groups"],
                                   w["q_group_map"],
                                   none_tensor,
                                   none_tensor,
                                   none_tensor,
                                   w.get("bias", none_tensor),
                                   temp_dq,
                                   max_dq_rows)

    # GPTQ

    elif "qweight" in w:

        if prescale != 1: w["scales"] *= prescale
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
                                       none_tensor,
                                       w["qzeros"],
                                       w["scales"],
                                       w["g_idx"].cpu(),
                                       w.get("bias", none_tensor),
                                       temp_dq,
                                       max_dq_rows)

        # GPTQ without g_idx

        else:

            return ext_c.make_q_matrix(w["qweight"],
                                       none_tensor,
                                       none_tensor,
                                       none_tensor,
                                       none_tensor,
                                       none_tensor,
                                       none_tensor,
                                       w["qzeros"],
                                       w["scales"],
                                       none_tensor,
                                       w.get("bias", none_tensor),
                                       temp_dq,
                                       max_dq_rows)


