#include "cudann.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

__global__ void
forward_d(f16* up, int num_up, f16* down, int num_down, f16* param) {
	int upI = blockDim.x * blockIdx.x + threadIdx.x;
	float v = 0;
	for (int downI = 0; downI < num_down; downI++) {
		float p = __half2float(param[upI*num_up+downI]);
		float d = __half2float(down[downI]);
		v += p * d;
	}
	v = (v>0) ? v : 0;
	up[upI] = __float2half_rn(v);
}

void forward(f16* up, int num_up, f16* down, int num_down, f16* param) {
	int threads = 256;
	int blocks = (num_up + threads - 1) / threads;
	forward_d<<<blocks, threads>>>(up, num_up, down, num_down, param);
}

void backward(f16* up, f16* up_err, int num_up, f16* down, f16* down_err, int num_down, f16* param) {
}

f16* alloc_f16_device(int count) {
	f16* f;
	cudaError_t err = cudaMalloc((void**)&f, count*sizeof(f16));
	if (err != cudaSuccess) {
		fprintf(stderr, "dalloc_f16(%d): %s", count, cudaGetErrorString(err));
		exit(1);
	}
	return f;
}

void memcpy_htod(f16* d, const f16* h, int count) {
	cudaError_t err = cudaMemcpy(d, h, count*sizeof(f16), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "memcpy_htod: %s", cudaGetErrorString(err));
		exit(1);
	}
	printf("hi\n");
}

void memcpy_dtoh(f16* h, const f16* d, int count) {
	cudaError_t err = cudaMemcpy(h, d, count*sizeof(f16), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		fprintf(stderr, "memcpy_dtoh: %s", cudaGetErrorString(err));
		exit(1);
	}
}
