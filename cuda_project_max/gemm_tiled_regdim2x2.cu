#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>





#define TILE_SIZE 16
#define PADDED_TILE_SIZE (TILE_SIZE + 1)

__global__ void sgemm_tiled_correct(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    __shared__ float As[PADDED_TILE_SIZE][PADDED_TILE_SIZE]; //保证tx + 1， ty + 1 不会越界
    __shared__ float Bs[PADDED_TILE_SIZE][PADDED_TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // Coordinates within shared memory for 2x2 register blocking
    int sm_row = ty * 2;
    int sm_col = tx * 2;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    // Each thread computes a 2x2 sub-block
    float c00 = 0.0f, c01 = 0.0f;
    float c10 = 0.0f, c11 = 0.0f;

    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE; // Number of tiles along K dimension, ceil division to cover all elements
    for (int t = 0; t < num_tiles; ++t){
        //Step1:Load tiles from global memory to shared memory(2x2 elements per thread for register blocking)
        //Step1.1:loading As
        int r0 = row;
        int r1 = row + 1;
        int c0 = t * TILE_SIZE + sm_col;
        int c1 = c0 + 1;
        As[sm_row][sm_col] = (r0<M && c0<K) ? A[r0 * K + c0] : 0.0f;
        As[sm_row][sm_col + 1] = (r0<M && c1<K) ? A[r0 * K + c1] : 0.0f;
        As[sm_row + 1][sm_col] = (r1<M && c0<K) ? A[r1 * K + c0] : 0.0f;
        As[sm_row + 1][sm_col + 1] =(r1<M && c1<K) ?  A[r1 * K + c1] : 0.0f;
        //Step1.2:loading Bs
        int br0 = t * TILE_SIZE + sm_row;
        int br1 = br0 + 1;
        int bc0 = col;
        int bc1 = bc0 + 1;
        Bs[sm_row][sm_col]  = (br0 < K && bc0 < N) ? B[br0 * N + bc0] : 0.0f;
        Bs[sm_row][sm_col +1]  = (br0 < K && bc1 < N) ? B[br0 * N + bc1] : 0.0f;
        Bs[sm_row + 1][sm_col]  = (br1 < K && bc0 < N) ? B[br1 * N + bc0] : 0.0f;
        Bs[sm_row + 1][sm_col + 1]  = (br1 < K && bc1 < N) ? B[br1 * N + bc1] : 0.0f;
        

        __syncthreads();

        //Step2:Calculation As*Bs (Accumulate into thread registers for 2x2 sub-block)
        for (int k_inner = 0; k_inner < TILE_SIZE; ++k_inner){
            float a0 = As[sm_row][k_inner];
            float a1 = As[sm_row + 1][k_inner];
            float b0 = Bs[k_inner][sm_col];
            float b1 = Bs[k_inner][sm_col + 1];
            c00 += a0 * b0;
            c01 += a0 * b1;
            c10 += a1 * b0;
            c11 += a1 * b1;
        }

        __syncthreads();
    }
    // -------------------------------
    // Step 3: Write results back to global memory
    // Each thread writes its 2x2 sub-block
    // -------------------------------
    if (row < M && col < N) C[row*N + col] = c00;
    if (row < M && col + 1 < N) C[row*N + col + 1] = c01;
    if (row + 1 < M && col < N) C[(row+1)*N + col] = c10;
    if (row + 1 < M && col+ 1 < N) C[(row+1)*N + col + 1] = c11;
    }

// Host
int main(){
    int M=1024, N=1024, K=1024;
    size_t sizeA = M*K*sizeof(float);
    size_t sizeB = K*N*sizeof(float);
    size_t sizeC = M*N*sizeof(float);

    float *hA=(float*)malloc(sizeA);
    float *hB=(float*)malloc(sizeB);
    float *hC=(float*)malloc(sizeC);

    for(int i=0;i<M*K;i++) hA[i]=1.0f;
    for(int i=0;i<K*N;i++) hB[i]=1.0f;

    float *dA,*dB,*dC;
    cudaMalloc(&dA,sizeA);
    cudaMalloc(&dB,sizeB);
    cudaMalloc(&dC,sizeC);

    cudaMemcpy(dA,hA,sizeA,cudaMemcpyHostToDevice);
    cudaMemcpy(dB,hB,sizeB,cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE/2,TILE_SIZE/2);
    dim3 grid((N+TILE_SIZE-1)/TILE_SIZE,(M+TILE_SIZE-1)/TILE_SIZE);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    sgemm_tiled_correct<<<grid,block>>>(dA,dB,dC,M,N,K);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms=0;
    cudaEventElapsedTime(&ms,start,stop);

    cudaMemcpy(hC,dC,sizeC,cudaMemcpyDeviceToHost);

    printf("C[0,0]=%f, Time=%f ms\n",hC[0],ms);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return 0;
}