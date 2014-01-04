#include "f16.h"

#include <stdio.h>
#include <immintrin.h>
#include <x86intrin.h>

void ftof16(f16* dst, const float* src, int count) {
	if (count%8 != 0) {
		fprintf(stderr, "f32tof16 needs multiple of 8 floats\n");
		exit(3);
	}
	for (int i = 0; i < count; i += 8) {
		__m256 f = _mm256_load_ps(&src[i]);
		__m128i h = _mm256_cvtps_ph(f, 0);
		*(__m128i*)&dst[i] = h;
	}
}

void f16tof(float* dst, const f16* src, int count) {
	if (count%8 != 0) {
		fprintf(stderr, "f32tof16 needs multiple of 8 floats\n");
		exit(3);
	}
	for (int i = 0; i < count; i+= 8) {
		__m128i h = *(__m128i*)&src[i];
		__m256 f = _mm256_cvtph_ps(h);
		_mm256_store_ps(&dst[i], f);
	}
}
