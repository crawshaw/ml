#include "cudann.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define count 16

int main() {
	float src[count];
	src[0] = 0.0;
	src[1] = 1.1;
	src[2] = 2.2;
	src[3] = 3.3;
	src[4] = 4.4;
	src[5] = 5.5;
	src[6] = 6.6;
	src[7] = 7.7;
	src[8] = 8.8;
	src[9] = 9.9;
	src[10] = 10.1;
	src[11] = 11.11;
	src[12] = 12.12;
	src[13] = 13.13;
	src[14] = 14.14;
	src[15] = 15.15;

	f16* d = (f16*)calloc(count, sizeof(f16));
	f32* dst = (f32*)valloc(count*sizeof(float));
	if (d == NULL || dst == NULL) {
		printf("d or dst == NULL");
		exit(2);
	}
	f32tof16(d, src, count);
	f16tof32(dst, d, count);

	int cmp = 0;
	for (int i = 0; i < count; i++) {
		if (src[i] - dst[i] > 0.01) {
			printf("i=%d, src=%.2f, dst=%.2f\n", i, src[i], dst[i]);
			cmp = 1;
		}
	}
	if (cmp != 0) {
		printf("src != dst\n");
		exit(2);
	}
}
