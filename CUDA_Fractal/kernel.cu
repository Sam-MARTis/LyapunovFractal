#include "kernel.h"
#define TX 32
#define TY 32

int divUp(int a, int b){
    return (a-1+b)/b;
}

__device__
unsigned char clip(int n){
    return n>255 ? 255:(n<0?0:n);
}

__global__
void lyapunovCalcKernel(uchar4 *d_out, bool *seq, float dx, float dy, int numIterations){
    
}

void lyapunovKernelLauncher(uchar4 *out, bool *sequence, int width, int height, int n, int m, int numIterations){
    uchar4 *d_out = 0;
    cudaMalloc(&d_out, n*m*sizeof(uchar4));
    float dx = ((float)width)/((float)n);
    float dy = ((float)height)/((float)m);
    lyapunovCalcKernel(d_out, sequence, dx, dy, numIterations);
    memccpy(out, d_out, n*m*sizeof(uchar4), cudaMemcpyDeviceToHost);
    cudaFree(d_out)
}


