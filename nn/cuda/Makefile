NVCC := /Developer/NVIDIA/CUDA-5.5/bin/nvcc
CCFLAGS := -ccbin /usr/bin/clang -arch=sm_30 -Xcompiler '-fPIC'
LDFLAGS := -L/Developer/NVIDIA/CUDA-5.5/lib -lcudart

cudann.so: cudann.o link.o f16.o
	g++ $(LDFLAGS) -shared -o $@ $+

cudann.o: cudann.cu
	$(NVCC) $(CCFLAGS) -dc -o $@ -c $<

link.o: cudann.o
	$(NVCC) $(CCFLAGS) -dlink -o $@ $<

f16.o: f16.c
	cc -mavx -mf16c -I. -c -o $@ $<

f16_test: f16_test.c f16.o
	cc -c -o f16_test.o f16_test.c
	cc -o $@ f16_test.o f16.o

simple.o: simple.c
	cc -I. -c -o $@ $<

simple: cudann.so simple.o
	cc -I. -o $@ $+

test: f16_test
	./f16_test

clean:
	rm -f *.so *.o simple *_test
