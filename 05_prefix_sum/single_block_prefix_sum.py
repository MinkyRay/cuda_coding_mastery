from numba import cuda
import numpy as np

###所以我们要根据数据的大小和结构 来设计blocks和threads，一定要匹配，
#num_blocks*threads_per_blocks数量可以大于等于总需要处理的数据量,或者是一个线程处理多个数据，
#这个就涉及到算法方面的改进了（while循环，用stride保证合并访问）
def test_single_block():
    """单block前缀和测试"""
    N = 256  # 只用一个block能处理的大小
    print(f"单block测试: N = {N}")
    
    input_cpu = np.random.randn(N).astype(np.float32)
    cpu_output = np.cumsum(input_cpu)
    
    @cuda.jit
    def single_block_prefix_sum(input_data, output_data, N):
        tid = cuda.threadIdx.x
        bdim = cuda.blockDim.x
        
        s1 = cuda.shared.array(256, dtype=np.float32)
        s2 = cuda.shared.array(256, dtype=np.float32)
        
        # 加载
        if tid < N:
            s1[tid] = input_data[tid]
        else:
            s1[tid] = 0.0
        s2[tid] = 0.0
        
        cuda.syncthreads()
        
        # Hillis-Steele
        src, dst = s1, s2
        stride = 1
        
        while stride < bdim:
            dst[tid] = src[tid]
            if tid >= stride:
                dst[tid] += src[tid - stride]
            cuda.syncthreads()
            src, dst = dst, src
            stride *= 2
        
        if tid < N:
            output_data[tid] = src[tid]
    
    # 执行 - 修复这里的语法！
    input_gpu = cuda.to_device(input_cpu)
    output_gpu = cuda.device_array(N, dtype=np.float32)
    
    # 正确调用语法：kernel[grid, block](args)
    single_block_prefix_sum[1, 256](input_gpu, output_gpu, N)  # ✅ 正确
    cuda.synchronize()
    
    gpu_output = output_gpu.copy_to_host()
    
    print(f"输入前5个: {input_cpu[:5]}")
    print(f"CPU结果前5个: {cpu_output[:5]}")
    print(f"GPU结果前5个: {gpu_output[:5]}")
    print(f"是否正确: {np.allclose(cpu_output, gpu_output, rtol=1e-5)}")

test_single_block()