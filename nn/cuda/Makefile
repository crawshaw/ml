ifndef CUDAROOT
	$(error CUDAROOT not set; have you run setup.sh?)
endif
ifndef CUDALIB
	$(error CUDALIB not set; have you run setup.sh?)
endif

NVCC := $(CUDAROOT)/bin/nvcc
#CCFLAGS := -ccbin /usr/bin/clang TODO: needed on darwin?
CCFLAGS := -arch=sm_30 -Xcompiler '-fPIC'
LDFLAGS := -L$(CUDALIB) -lcudart

cudann.so: cudann.o link.o
	g++ $(LDFLAGS) -shared -o $@ $+

cudann.o: cudann.cu
	$(NVCC) $(CCFLAGS) -dc -o $@ -c $<

link.o: cudann.o
	$(NVCC) $(CCFLAGS) -dlink -o $@ $<

clean:
	rm -f *.so *.o
