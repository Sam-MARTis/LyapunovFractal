#include "kernel.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <sstream>
#include <string>
#include <iomanip>
#include <cstdint>
#define W 8.f
#define H 8.f
#define XMIN -0.2f
#define YMIN -0.2f
#define NX 128*16
#define NY 128*16


#define NUM_ITERATIONS 20000


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
        out.put(data[i].x); 
        out.put(data[i].y); 
        out.put(data[i].z); 

    }

    out.close();
    std::cout << "Saved image to " << filename << "\n";
}

int main(){
    uchar4 *data = (uchar4*)malloc(NX*NY*sizeof(uchar4));
    char order[] = "AB";
    int charLen = sizeof(order)/sizeof(order[0]);
    bool *seq = (bool*)malloc(charLen * sizeof(bool));
    for (int i = 0; i < charLen; i++) {
        if (order[i] == 'A') {
            seq[i] = true;
        } else {
            seq[i] = false;
        }
    }


    std::ostringstream filename;
    filename << "seq-" << order
             << "_iters-" << NUM_ITERATIONS
             << std::fixed << std::setprecision(2)
             << "_xmin-" << XMIN
             << "_ymin-" << YMIN
             << "_w-" << W
             << "_h-" << H
             << "_nx-" << NX
             << "_ny-" << NY
             << ".ppm";

    std::string fname = filename.str(); 

    lyapunovKernelLauncher(data, seq, charLen, XMIN, YMIN, W, H, NX, NY, NUM_ITERATIONS);
    save_ppm(filename.str().c_str(), data, NX, NY);

    save_ppm(fname.c_str(), data, NX, NY);
    free(data);
    free(seq);
    return 0;
}