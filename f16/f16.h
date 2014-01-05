typedef unsigned short f16;

void ftof16(f16* dst, const float* src, int count);
void f16tof(float* dst, const f16* src, int count);

void f16add(f16* dst, f16* src, int len);
void f16sub(f16* dst, f16* src, int len);
void f16mul(f16* dst, int len, float val);
