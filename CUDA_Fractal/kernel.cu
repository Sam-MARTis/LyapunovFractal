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
    if((xIdx>=n) || (yIdx>=m)) return;
    const int idx = xIdx + yIdx*n;

    const int SL = seqLen;
    const float A = ((float)xIdx)*dx;
    const float B = ((float)yIdx)*dy;
    int seqIterator = 0;
    float seqArr[SL] = {0.0};
    for(int i=0; i<SL; i++){
        if(seq[i]){
            seqArr[i] = A;
        }
        else{
            seqArr[i] = B;
        }
    }
    
    curandState state;
    curand_init(seed, xIdx, yIdx, &state);
    float x = curand_uniform(&state)*0.9999; // To handle edge case when curand returns 1
    float sum=x;
    int j = 0;
    for(int i=0; i<numIterations; i++){
        
        x = seqArr[j]*x*(1-x);
        sum += log(abs(seqArr[j]*(1-2*x)));
        j = (j+1)%SL;
    }
    const float lyapunovExponent =sum/((float)SL);
    
    if(lyapunovExponent<0){
        const int intensity = clip(round(-lyapunovExponent*255.0f));
        d_out[idx].x = clip(255 - intensity);
        d_out[idx].y = clip(255-intensity);
        d_out[idx].z = 0;
    }
    else{
        const int intensity = clip(round(lyapunovExponent*255.0f));
        d_out[idx].x = 0;
        d_out[idx].y = 0;
        d_out[idx].z = clip(255-intensity);
    }

    d_out[idx].w = 1;





    

}



void lyapunovKernelLauncher(uchar4 *out, bool *sequence, int sequenceLength, int width, int height, int n, int m, int numIterations){
    uchar4 *d_out = 0;
    bool *d_seq = 0;
    cudaMalloc(&d_seq, sequenceLength*sizeof(bool));
    cudaMemcpy(d_seq, sequence, sequenceLength*sizeof(bool), cudaMemcpyHostToDevice);
    cudaMalloc(&d_out, n*m*sizeof(uchar4));
    float dx = ((float)width)/((float)n);
    float dy = ((float)height)/((float)m);
    const dim3 blockSize(TX, TY);
    const dim3 gridSize(divUp(n, TX), divUp(m, TY));

    lyapunovCalcKernel<<<gridSize, blockSize>>>(d_out, d_seq, sequenceLength,dx, dy, n, m, numIterations);
    cudaMemcpy(out, d_out, n*m*sizeof(uchar4), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    cudaFree(d_seq);
}


