import time
from functools import partial
from typing import Optional

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="elementwise_add_lib",
    sources=["elementwise_add.cu", "torch_bindings.cu"],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++20"],
)


def run_benchmark(
    func: callable,
    a: torch.Tensor,
    b: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 1000,
    show_all: bool = False,
):
    if out is not None:
        out.fill_(0)
    # warmup
    if out is not None:
        for i in range(warmup):
            func(a, b, out)
    else:
        for i in range(warmup):
            _ = func(a, b)
    torch.cuda.synchronize()

    start = time.time()
    if out is not None:
        for i in range(iters):
            func(a, b, out)
    else:
        for i in range(iters):
            _ = func(a, b)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters

    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:2]
    out_val = [round(v, 8) for v in out_val]
    print(f"{out_info:>18}: {out_val}, time:{mean_time:.8f}ms")
    if show_all:
        print(out)
    return out, mean_time


S = [1024, 2048, 4096]
K = [1024, 2048, 4096]
SK = [(s, k) for s in S for k in K]

for s, k in SK:
    print(f"Benchmarking element-wise add with size: ({s}, {k})")
    a = torch.randn((s, k)).cuda().float().contiguous()
    b = torch.randn((s, k)).cuda().float().contiguous()
    out = torch.zeros_like(a).cuda().float().contiguous()
    run_benchmark(lib.elementwise_add_f32, a, b, "f32", out)
    run_benchmark(lib.elementwise_add_f32x4, a, b, "f32x4", out)
    run_benchmark(
        lambda a, b, out: torch.add(a, b, out=out), a, b, "torch.add f32", out
    )

    print("-" * 85)
    a_f16 = a.half().contiguous()
    b_f16 = b.half().contiguous()
    out_f16 = out.half().contiguous()
    run_benchmark(lib.elementwise_add_f16, a_f16, b_f16, "f16", out_f16)
    run_benchmark(lib.elementwise_add_f16x2, a_f16, b_f16, "f16x2", out_f16)
    run_benchmark(lib.elementwise_add_f16x8, a_f16, b_f16, "f16x8", out_f16)
    run_benchmark(lib.elementwise_add_f16x8_pack, a_f16, b_f16, "f16x8pack", out_f16)
    run_benchmark(
        lambda a, b, out: torch.add(a, b, out=out), a_f16, b_f16, "f16_th", out_f16
    )
    print("-" * 85)
