UNAME_S := $(shell uname)

ifeq ($(UNAME_S), Darwin)
	LDFLAGS = -Xlinker -framework,OpenGL, -XLinker - framework, GLUT
else
	LDFLAGS += -lcuda -lcudart
	LDFLAGS += -lglut -lGL -lGLU -lGLEW
endif



NVCC = /usr/local/cuda-12.8/bin/nvcc
NVCC_FLAGS += -g -G -Xcompiler -Wall
OBJS = main.o kernel.o

all: main.exe


main.exe: main.o kernel.o
	$(NVCC) $^ -o $@ -Wno-deprecated-gpu-targets $(LDFLAGS)

main.o: main.cpp kernel.h 
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ 

kernel.o : kernel.cu kernel.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ 
	
clean:
	rm -f $(OBJS) main.exe