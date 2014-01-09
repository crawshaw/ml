#ifdef __cplusplus
extern "C" {
#endif

#include "../../f16/f16.h"

void forward(f16* up, int num_up, f16* down, int num_down, f16* param);
void backward(f16* up, f16* up_err, int num_up, f16* down, f16* down_err, int num_down, f16* param);

void f16devsub(f16* dst, f16* a, f16* b, int count);
f16* alloc_f16_device(int count);
void memcpy_htod(f16* d, const f16* h, int count);
void memcpy_dtoh(f16* h, const f16* d, int count);

#ifdef __cplusplus
}  // extern C
#endif
