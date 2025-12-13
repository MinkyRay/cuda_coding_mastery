\# CUDA GEMM Optimization Practice



This repository contains a small set of CUDA GEMM (SGEMM) implementations,

written for learning and performance exploration purposes.



The goal is to understand how different optimization techniques affect

performance, starting from a naive baseline and gradually introducing

shared memory and register-level blocking.



---



\## Implementations



\### 1. `gemm\_naive.cu`

\- Each thread computes one element of C

\- Directly loads A and B from global memory

\- No shared memory, no tiling

\- Serves as a correctness and performance baseline



\### 2. `gemm\_tiled.cu`

\- Uses shared memory tiling

\- Each thread computes one output element

\- Shared memory padding is applied to avoid bank conflicts

\- Significantly reduces global memory traffic



\### 3. `gemm\_tiled\_regdim2x2.cu`

\- Shared memory tiling + register blocking

\- Each thread computes a 2×2 output tile

\- Improves arithmetic intensity and reduces instruction overhead

\- Demonstrates noticeable performance improvement over the 1×1 tiled version



---



\## Key Techniques Covered



\- Global memory vs shared memory access patterns

\- Shared memory tiling

\- Shared memory padding to avoid bank conflicts

\- Register blocking (2×2 per thread)

\- Boundary handling for non-multiple tile sizes

\- Performance measurement using CUDA events



---



\## Build \& Run



Example (Windows + NVCC):



```bash

nvcc gemm\_naive.cu -o gemm\_naive

nvcc gemm\_tiled.cu -o gemm\_tiled

nvcc gemm\_tiled\_regdim2x2.cu -o gemm\_reg2x2



