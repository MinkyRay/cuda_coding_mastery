#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> 

#define TILE_SIZE 16
// PADDING method：TILE_SIZE + 1 to avoid "Shared Memory Bank Conflict"
#define PADDED_TILE_SIZE (TILE_SIZE + 1)


// --------------------------------------------------------
// CUDA Kernel: Tiled GEMM with Shared Memory Padding (V3)
// --------------------------------------------------------
__global__ void sgemm_tiled_padded(float *A, float *B, float *C, int M, int N, int K) {
    


    __shared__ float As[TILE_SIZE][PADDED_TILE_SIZE];// TILE_SIZE x PADDED_TILE_SIZE: pad 1 to avoid bank conflict in column access
    __shared__ float Bs[TILE_SIZE][PADDED_TILE_SIZE];

    // thread index
    int tx = threadIdx.x; // thread column within block
    int ty = threadIdx.y; // thread row within block

    // global index in C
    int row = blockIdx.y * TILE_SIZE + ty;  // row in C
    int col = blockIdx.x * TILE_SIZE + tx; // col in C

    // C[row, col] accumulate register
    float Cvalue = 0.0f;


    // MAIN LOOP：Block Accumulation Loop

    int num_tiles = K / TILE_SIZE; 
    
    //For Loop to accumulate blook result
    for (int t = 0; t < num_tiles; ++t){
        //Step1: loading data from Global memory to Shared memory
        //Step1.1: A to As
        int global_a_row = row;
        int global_a_col = t * TILE_SIZE + tx; // coalesced memory access along row
        if (global_a_row < M && global_a_col < K){
            As[ty][tx] = A[global_a_row*K + global_a_col]; // linearized 1D access
        }
        else{
            As[ty][tx] = 0.0f; // padding for boundary
        }

        //Step1.2: B to Bs
        int global_b_row = t * TILE_SIZE + ty;
        int global_b_col = col;// blockIdx.x * TILE_SIZE + tx : coalesced along row
        if (global_b_row < K && global_b_col < N){
            Bs[ty][tx] = B[global_b_row*N + global_b_col];
        }
        else{
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        //Step2: Calculation As*Bs
        for (int k_inner = 0; k_inner < TILE_SIZE; ++k_inner){
            Cvalue += As[ty][k_inner] * Bs[k_inner][tx]; // As row contiguous for reuse, Bs padded for bank conflict avoidance
        }

        __syncthreads();
    }

    if (row < M && col < N){
        C[row*N + col] = Cvalue; // write final result
    }
}



// Host Code
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

    // Launch configuration: Block Size = TILE_SIZE x TILE_SIZE
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    
    // Grid Size
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, 
                  (M + TILE_SIZE - 1) / TILE_SIZE);


    // Begin V3 Tiled Kernel

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Computing... Grid size: (%d, %d)\n", gridSize.x, gridSize.y);

    cudaEventRecord(start);
    
    //V3 Tiled Kernel
    sgemm_tiled_padded<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);

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
//This kernel demonstrates correctness of shared-memory tiling, but does not outperform naive kernel on RTX 4050 for problem size 1024³ due to synchronization overhead and strong cache behavior of naive kernel.