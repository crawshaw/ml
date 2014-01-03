#ifdef __cplusplus
extern "C" {
#endif

typedef float f32;
typedef unsigned short f16;

void forward(f16* up, int num_up, f16* down, int num_down, f16* param);
void backward(f16* up, f16* up_err, int num_up, f16* down, f16* down_err, int num_down, f16* param);

f16* alloc_f16_device(int count);
void memcpy_htod(f16* d, const f16* h, int count);
void memcpy_dtoh(f16* h, const f16* d, int count);

void f32tof16(f16* dst, const f32* src, int count);
void f16tof32(f32* dst, const f16* src, int count);

#ifdef __cplusplus
}  // extern C
#endif
