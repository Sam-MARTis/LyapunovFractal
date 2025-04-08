#ifndef KERNEL_H
#define KERNEL_H
struct uchar4;

void lyapunovKernelLauncher(uchar4 *out, bool *sequence, int seqLength,float xstart, float ystart, float width, float height, int n, int m, int numIterations);

#endif