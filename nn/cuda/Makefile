NVCC := /Developer/NVIDIA/CUDA-5.5/bin/nvcc
CCFLAGS := -ccbin /usr/bin/clang -arch=sm_30 -Xcompiler '-fPIC'
LDFLAGS := -L/Developer/NVIDIA/CUDA-5.5/lib -lcudart

cudann.so: cudann.o link.o
	g++ $(LDFLAGS) -shared -o $@ $+

cudann.o: cudann.cu
	$(NVCC) $(CCFLAGS) -dc -o $@ -c $<

link.o: cudann.o
	$(NVCC) $(CCFLAGS) -dlink -o $@ $<

clean:
	rm -f *.so *.o
