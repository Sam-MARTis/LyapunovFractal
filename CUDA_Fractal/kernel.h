#ifndef KERNEL_H
#define KERNEL_H
struct uchar4;

void lyapunovKernelLauncher(uchar4 *out, bool *sequence, int width, int height, int n, int m, int numIterations);

#endif