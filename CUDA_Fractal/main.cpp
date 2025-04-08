#include "kernel.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cstdint>
#define W 16.0f
#define H 16.0f
#define NX 128*16
#define NY 128*16
#define SEQUENCE_LENGTH 2

#define NUM_ITERATIONS 2000


void save_pgm(const char *filename, unsigned char *data, int width, int height) {
    FILE *f = fopen(filename, "wb");
    if (!f) {
        perror("Error opening file");
        return;
    }
    fprintf(f, "P5\n%d %d\n255\n", width, height);
    fwrite(data, sizeof(unsigned char), width * height, f);

    fclose(f);
}

void save_ppm(const char* filename, uchar4* data, int width, int height) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open file.\n";
        return;
    }


    out << "P6\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height; ++i) {
        out.put(data[i].x);  // R
        out.put(data[i].y);  // G
        out.put(data[i].z);  // B

    }

    out.close();
    std::cout << "Saved image to " << filename << "\n";
}

int main(){
    uchar4 *data = (uchar4*)malloc(NX*NY*sizeof(uchar4));
    bool *seq = (bool*)malloc(2 * sizeof(bool));
    seq[0] = false;
    seq[1] = true;

    lyapunovKernelLauncher(data, seq, SEQUENCE_LENGTH,  W, H, NX, NY, NUM_ITERATIONS);
    
    save_ppm("result.ppm", data, NX, NY);
    free(data);
    free(seq);
    return 0;
}