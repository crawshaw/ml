#include "cudann.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

__device__ float sigmoid(float x) {
	return 1.0 / (1.0 + expf(-x));
}

__device__ float sigmoid_gradient(float x) {
	return x * (1 - x);
}

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
	//v = sigmoid(v);
	up[upI] = __float2half_rn(v);
}

void forward(f16* up, int num_up, f16* down, int num_down, f16* param) {
	int threads = 8;
	int blocks = (num_up + threads - 1) / threads;
	forward_d<<<blocks, threads>>>(up, num_up, down, num_down, param);
}

__global__ void
backward_d(f16* up, f16* up_err, int num_up, f16* down, f16* down_err, int num_down, f16* param) {
	int downI = blockDim.x * blockIdx.x + threadIdx.x;

	float v = 0;
	for (int upI = 0; upI < num_up; upI++) {
		int paramI = upI*num_up+downI;
		float p = __half2float(param[paramI]);
		float u = __half2float(up_err[upI]);
		v += p * u;
	}
	down_err[downI] = __float2half_rn(v);

	float down_orig = __half2float(down[downI]);
	for (int upI = 0; upI < num_up; upI++) {
		float up_orig = __half2float(up[upI]);
		//float gradient = sigmoid_gradient(up_orig);
		float gradient = 1;
		float delta = __half2float(up_err[upI]) * gradient * down_orig;
		int paramI = upI*num_up+downI;
		float p = __half2float(param[paramI]);
		p = (p>0) ? p : 0; // max(0, v)
		param[paramI] = __float2half_rn(p + delta);
	}
}

void backward(f16* up, f16* up_err, int num_up, f16* down, f16* down_err, int num_down, f16* param) {
	int threads = 256;
	int blocks = (num_down + threads - 1) / threads;
	backward_d<<<blocks, threads>>>(up, up_err, num_up, down, down_err, num_down, param);
}

__global__ void f16devsub_d(f16* dst, f16* a, f16* b, int count) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	float af = __half2float(a[i]);
	float bf = __half2float(b[i]);
	dst[i] = __float2half_rn(af - bf);
}

void f16devsub(f16* dst, f16* a, f16* b, int count) {
	int threads = 128;
	int blocks = (count + threads - 1) / threads;
	f16devsub_d<<<blocks, threads>>>(dst, a, b, count);
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
