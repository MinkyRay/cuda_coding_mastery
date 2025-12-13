#include <cuda_runtime.h>
#include <stdio.h>


__global__ void sgemm_naive(float *A, float *B, float *C, int M, int N, int K) {

    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;


    if (row < M && col < N) {
        float tmp = 0.0f;
        for (int k = 0; k < K; ++k) {
            // ------------------------------------------------------------------
            // [Bottleneck Location: High Global Memory Traffic within the Loop]
            // Bottleneck: Excessive Global Memory (DRAM) accesses inside the inner loop.
            // 
            // - Arithmetic Intensity (AI) is extremely low: ~1/4 FLOPs/Byte. 
            //   The kernel is severely MEMORY BOUND.
            // - Data reuse is poor: For each C[row, col] calculation, 2*K single-precision floats 
            //   are fetched from the slow Global Memory.
            // - Although B access is coalesced, A access is broadcast, wasting time waiting for data.
            // ------------------------------------------------------------------
            tmp += A[row * K + k] * B[k * N + col];
        }

       
        C[row * N + col] = tmp; // write final result
    }
}


// Host Code: 主函数


int main() {
    int M = 1024;
    int N = 1024;
    int K = 1024;
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);


    float *h_A = (float*)malloc(bytes_A);
    float *h_B = (float*)malloc(bytes_B);
    float *h_C = (float*)malloc(bytes_C);
    

    for(int i=0; i<M*K; i++) h_A[i] = 1.0f;
    for(int i=0; i<K*N; i++) h_B[i] = 1.0f;


    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);
    
    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);

    dim3 blockSize(32, 32);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 
                  (M + blockSize.y - 1) / blockSize.y);




    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Computing... Grid size: (%d, %d)\n", gridSize.x, gridSize.y);


    cudaEventRecord(start);

    //Begin kernel
    sgemm_naive<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);


    cudaEventRecord(stop);
    

    cudaEventSynchronize(stop);


    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // ==========================================
    
    cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost);
    

    printf("Result top-left: %f\n", h_C[0]);
    printf("Time elapsed: %f ms\n", milliseconds);
    

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}



//Naive Kernel Performance Bottleneck AnalysisBottleneck Type: Memory BoundThe performance of this Naive GEMM Kernel is critically low, achieving less than 1% of the GPU's theoretical peak performance (based on the 3.48ms runtime). The primary bottleneck is the inefficient and excessive reliance on Global Memory (DRAM) access.Extremely Low Arithmetic Intensity (AI):Inside the critical inner loop, for every two floating-point operations (one multiply, one add), the kernel must fetch two single-precision floats (8 Bytes) from Global Memory.$$\text{AI} \approx \frac{2 \text{ FLOPs}}{8 \text{ Bytes}} = 0.25 \text{ FLOPs/Byte}$$This ratio is significantly lower than the computational throughput of modern GPUs, leaving the processing cores MEMORY BOUND and stalled (Data Starvation) waiting for data.Poor Data Locality and Reuse:Each thread performs $K$ loop iterations, repeatedly accessing the same row of matrix $A$ and the same column of matrix $B$. The lack of explicit local data storage means high latency Global Memory is accessed in every iteration, drastically limiting throughput.Inefficient Memory Access Pattern:While accessing matrix $B$ exhibits highly efficient Coalesced Access, the access pattern for matrix $A$ results in a Broadcast scenario where 32 threads read the exact same element. While not a conflict, this still represents an inefficient use of the massive available memory bandwidth for overall computation.Optimization Goal: The subsequent optimization must implement Shared Memory Tiling to transform the complexity of memory access from $O(N^3)$ to $O(N^2)$, dramatically increasing data locality and kernel throughput.
