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

void f16add(f16* dst, f16* src, int len) {
	if (len%8 != 0) {
		fprintf(stderr, "f16add needs multiple of 8 floats\n");
		exit(3);
	}

	for (int i = 0; i < len; i+= 8) {
		__m256 srcf = _mm256_cvtph_ps(*(__m128i*)&src[i]);
		__m256 dstf = _mm256_cvtph_ps(*(__m128i*)&dst[i]);
		dstf = _mm256_add_ps(dstf, srcf);
		*(__m128i*)&dst[i] = _mm256_cvtps_ph(dstf, 0);
	}
}

void f16sub(f16* dst, f16* src, int len) {
	if (len%8 != 0) {
		fprintf(stderr, "f16sub needs multiple of 8 floats\n");
		exit(3);
	}

	for (int i = 0; i < len; i+= 8) {
		__m256 srcf = _mm256_cvtph_ps(*(__m128i*)&src[i]);
		__m256 dstf = _mm256_cvtph_ps(*(__m128i*)&dst[i]);
		dstf = _mm256_sub_ps(dstf, srcf);
		*(__m128i*)&dst[i] = _mm256_cvtps_ph(dstf, 0);
	}
}

void f16mul(f16* dst, int len, float val) {
	if (len%8 != 0) {
		fprintf(stderr, "f16mul needs multiple of 8 floats\n");
		exit(3);
	}

	__m256 valvec = _mm256_set1_ps(val);
	for (int i = 0; i < len; i+= 8) {
		__m256 dstf = _mm256_cvtph_ps(*(__m128i*)&dst[i]);
		dstf = _mm256_mul_ps(dstf, valvec);
		*(__m128i*)&dst[i] = _mm256_cvtps_ph(dstf, 0);
	}
}
