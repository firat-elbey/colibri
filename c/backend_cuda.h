#ifndef COLIBRI_BACKEND_CUDA_H
#define COLIBRI_BACKEND_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define COLI_CUDA_MAX_DEVICES 16

/* Opaque, persistent device copy of one resident quantized tensor. */
typedef struct ColiCudaTensor ColiCudaTensor;

/* Devices are CUDA ordinals, not positions in the input list. */
int coli_cuda_init(const int *devices, int count);
void coli_cuda_shutdown(void);
int coli_cuda_device_count(void);
int coli_cuda_device_at(int index);
int coli_cuda_mem_info(int device, size_t *free_bytes, size_t *total_bytes);
/* device < 0 returns aggregate statistics for all configured devices. */
void coli_cuda_stats(int device, size_t *tensor_count, size_t *tensor_bytes);

/* Upload without executing, so capacity failures happen during model startup. */
int coli_cuda_tensor_upload(ColiCudaTensor **tensor,
                            const void *weights, const float *scales,
                            int fmt, int I, int O, int device);

/*
 * y[S,O] = x[S,I] @ W[O,I]^T.
 * fmt matches QT in glm.c: 0=f32, 1=int8, 2=int4, 3=int2.
 * The first successful call uploads W and its row scales; later calls reuse it.
 * Returns 1 on success and 0 when CUDA is not initialized or the format is invalid.
 */
int coli_cuda_matmul(ColiCudaTensor **tensor,
                     float *y, const float *x,
                     const void *weights, const float *scales,
                     int fmt, int S, int I, int O, int device);

/*
 * Synchronous matmul for weights NOT resident in VRAM: streams them through a
 * persistent per-device scratch slot (no ColiCudaTensor is created). Pays PCIe
 * for the weights on every call, so it is only worth it when S is large enough
 * to amortize the transfer (prefill, MTP verification batches).
 */
int coli_cuda_matmul_stream(float *y, const float *x,
                            const void *weights, const float *scales,
                            int fmt, int S, int I, int O, int device);

/*
 * Enqueue a full expert FFN — hh = W_d · (silu(W_g·x) * (W_u·x)) — on the
 * device stream of the three resident tensors and return immediately, so the
 * CPU can compute other experts while the GPU works. Rows rows[0..nr) are
 * gathered from x (row stride = W_g input dim) into pinned staging. After
 * coli_cuda_sync(device), *hh_out points at pinned host memory with the nr
 * result rows; it stays readable until the next enqueue that follows the sync.
 * All three tensors must live on the same device. Returns 0 on any failure
 * (caller falls back to the CPU path; partially enqueued work only touches
 * scratch buffers and is harmless).
 */
int coli_cuda_ffn_enqueue(ColiCudaTensor *g, ColiCudaTensor *u, ColiCudaTensor *d,
                          const float *x, const int *rows, int nr, float **hh_out);

/*
 * Wait for all work enqueued on `device` (all configured devices when
 * device < 0) and reset the transient staging arenas.
 */
int coli_cuda_sync(int device);

void coli_cuda_tensor_free(ColiCudaTensor *tensor);
size_t coli_cuda_tensor_bytes(const ColiCudaTensor *tensor);
int coli_cuda_tensor_device(const ColiCudaTensor *tensor);

#ifdef __cplusplus
}
#endif

#endif
