#include "kernel.h"
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <iostream>

#define TX 32
#define TY 16
#define seed 42

int divUp(int a, int b){
    return (a-1+b)/b;
}

__device__
unsigned char clip(int n){
    return n>255 ? 255:(n<0?0:n);
}

__global__
void lyapunovCalcKernel(uchar4 *d_out, bool *seq, int seqLen, float dx, float dy, int n, int m, float width, float height, int numIterations){
    // printf("Kernel launched");
    const int xIdx = threadIdx.x + blockDim.x*blockIdx.x;
    const int yIdx = threadIdx.y + blockDim.y*blockIdx.y;
    if((xIdx>=n) || (yIdx>=m)) return;
    const int idx = xIdx + yIdx*n;
    

    const int SL = seqLen;
    const float A = ((float)xIdx)*dx - width/2;
    const float B = ((float)yIdx)*dy - height/2;
    float seqArr[2] = {0.0};
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
    // float x = curand_uniform(&state);
    float sum=x;
    int j = 0;
    // printf("seqArr[0] = %f\n", seqArr[0]);
    // printf("seqArr[1] = %f\n", seqArr[1]);
    for(int i=0; i<numIterations/20; i++ ){
        x = seqArr[j]*x*(1-x);
        j = (j+1)%SL;
    }
    for(int i=0; i<numIterations; i++){
        
        x = seqArr[j]*x*(1-x);
        sum += log(abs(seqArr[j]*(1-2*x)));
        if(sum==INFINITY){
            // printf("Sum is infinity\n");
            // break;
            return;
        }
        j = (j+1)%SL;
    }
    // printf("Sum: %f\n", sum);
    // printf("SL: %d\n", SL);
    const float lyapunovExponent = sum/((float)numIterations);
    // printf("Lyapunov exponent at (%d, %d): %f\n", xIdx, yIdx, lyapunovExponent);

    // printf("Lyapunov exponent at (%d, %d): %f\n", xIdx, yIdx, lyapunovExponent);
    if(lyapunovExponent<0){
        const int intensity = clip(round(255* (-lyapunovExponent)));
        d_out[idx].x = clip(255 - intensity);
        d_out[idx].y = clip(255-intensity);
        d_out[idx].z = 10;
    } 
    else{
        const int intensity = clip(round(255* (lyapunovExponent)));
        d_out[idx].x = 10;
        d_out[idx].y = 10;
        d_out[idx].z = clip(255-intensity);
    }

    d_out[idx].w = 255;





    

}



void lyapunovKernelLauncher(uchar4 *out, bool *sequence, int sequenceLength, float width, float height, int n, int m, int numIterations){
    printf("Kernel launcher called");
    uchar4 *d_out = 0;
    bool *d_seq = 0;
    cudaMalloc(&d_seq, sequenceLength*sizeof(bool));
    cudaMemcpy(d_seq, sequence, sequenceLength*sizeof(bool), cudaMemcpyHostToDevice);
    cudaMalloc(&d_out, n*m*sizeof(uchar4));
    float dx = (width)/((float)n);
    float dy = (height)/((float)m);
    const dim3 blockSize(TX, TY);
    const dim3 gridSize(divUp(n, TX), divUp(m, TY));
    // const dim3 gridSize(20, 20, 1);

    printf("About to call kernel\n");

    lyapunovCalcKernel<<<gridSize, blockSize>>>(d_out, d_seq, sequenceLength,dx, dy, n, m, width, height, numIterations);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << "\n";
    }
    cudaDeviceSynchronize();
    printf("Synchronized\n");
    cudaMemcpy(out, d_out, n*m*sizeof(uchar4), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    cudaFree(d_seq);
}


