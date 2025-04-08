#include "kernel.h"
#include <curand.h>
#include <curand_kernel.h>

#define TX 32
#define TY 32
#define seed 42

int divUp(int a, int b){
    return (a-1+b)/b;
}

__device__
unsigned char clip(int n){
    return n>255 ? 255:(n<0?0:n);
}

__global__
void lyapunovCalcKernel(uchar4 *d_out, bool *seq, int seqLen, float dx, float dy, int n, int m, int numIterations){
    const int xIdx = threadIdx.x + blockDim.x*blockIdx.x;
    const int yIdx = threadIdx.y + blockDim.y*blockIdx.y;
    const float A = ((float)xIdx)*dx;
    const float B = ((float)yIdx)*dy;
    int seqIterator = 0;
    float seqArr[seqLen] = {0.0};
    for(int i=0; i<seqLen; i++){
        if(seq[i]){
            seqArr[i] = A;
        }
        else{
            seqArr[i] = B;
        }
    }
    float sum=0;
    curandState state;
    curand_init(seed, xIdx, yIdx, &state);
    float x = curand_uniform(&state)*0.99;

    

}



void lyapunovKernelLauncher(uchar4 *out, bool *sequence, int sequenceLength, int width, int height, int n, int m, int numIterations){
    uchar4 *d_out = 0;
    bool *d_seq = 0;
    cudaMalloc(&d_seq, sequenceLength*sizeof(bool));
    cudaMalloc(&d_out, n*m*sizeof(uchar4));
    float dx = ((float)width)/((float)n);
    float dy = ((float)height)/((float)m);
    const dim3 blockSize(TX, TY);
    const dim3 gridSize(divUp(n, TX), divUp(m, TY));

    lyapunovCalcKernel<<<gridSize, blockSize>>>(d_out, d_seq, sequenceLength,dx, dy, n, m, numIterations);
    memccpy(out, d_out, n*m*sizeof(uchar4), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
}


