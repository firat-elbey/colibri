#include "backend_cuda.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

/*
 * CUDA backend, stage 2.
 *
 * Three execution paths, one stream per device:
 *   1. coli_cuda_matmul      — synchronous, weights resident in VRAM (dense tier).
 *   2. coli_cuda_ffn_enqueue — ASYNC expert FFN on pinned experts: the call returns
 *      immediately and the CPU computes other experts while the GPU works; results
 *      are collected after coli_cuda_sync. This is what lets the VRAM tier ADD its
 *      bandwidth instead of serializing into the token's critical path.
 *   3. coli_cuda_matmul_stream — synchronous, weights NOT resident: they stream
 *      through a persistent scratch slot. Pays PCIe per call, so it is only worth
 *      it for large batches (prefill), where the transfer amortizes over S rows.
 *
 * Kernels dequantize on use (same container semantics as the CPU path) and
 * accumulate in f32. Rows whose byte stride allows it use vectorized uint32/float4
 * loads; odd shapes fall back to the scalar path (kept bit-compatible with the
 * original kernel).
 */

struct ColiCudaTensor {
    void *weights;
    float *scales;
    size_t weight_bytes;
    int fmt, I, O, device;
    int tracked;
};

/* Bump arena over a chain of blocks (pinned host or device memory). Blocks are
 * retained across resets and freed only at shutdown. Reset happens at
 * coli_cuda_sync: pointers handed out stay readable until the next enqueue. */
typedef struct ArenaBlock {
    void *p;
    size_t cap, off;
    struct ArenaBlock *next;
} ArenaBlock;

typedef struct {
    int device;
    cudaStream_t stream;
    ArenaBlock *pin, *dev;                 /* transient staging, reset on sync */
    void *wbuf; size_t wcap;               /* persistent scratch: streamed weights */
    float *wsc; size_t wsc_cap;            /* persistent scratch: streamed row scales */
    float *xbuf, *ybuf; size_t x_cap, y_cap; /* persistent scratch: activations */
    size_t tensor_count, tensor_bytes;
} DeviceContext;

static DeviceContext g_ctx[COLI_CUDA_MAX_DEVICES];
static int g_nctx;

static int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "[CUDA] %s: %s\n", what, cudaGetErrorString(err));
    return 0;
}

static DeviceContext *find_ctx(int device) {
    for (int i = 0; i < g_nctx; i++) if (g_ctx[i].device == device) return &g_ctx[i];
    return nullptr;
}

static int select_ctx(DeviceContext *ctx) {
    return ctx && cuda_ok(cudaSetDevice(ctx->device), "select device");
}

static size_t row_bytes(int fmt, int I) {
    if (fmt == 0) return (size_t)I * sizeof(float);
    if (fmt == 1) return (size_t)I;
    if (fmt == 2) return (size_t)(I + 1) / 2;
    if (fmt == 3) return (size_t)(I + 3) / 4;
    return 0;
}

/* ---------------- arenas ---------------- */

#define ARENA_MIN_BLOCK ((size_t)32 << 20)

static void *arena_take(ArenaBlock **list, size_t bytes, int pinned) {
    bytes = (bytes + 255) & ~(size_t)255;
    for (ArenaBlock *b = *list; b; b = b->next)
        if (b->cap - b->off >= bytes) { void *p = (char *)b->p + b->off; b->off += bytes; return p; }
    size_t cap = bytes > ARENA_MIN_BLOCK ? bytes : ARENA_MIN_BLOCK;
    ArenaBlock *b = (ArenaBlock *)std::calloc(1, sizeof(*b));
    if (!b) return nullptr;
    cudaError_t err = pinned ? cudaHostAlloc(&b->p, cap, cudaHostAllocDefault)
                             : cudaMalloc(&b->p, cap);
    if (!cuda_ok(err, pinned ? "pinned arena block" : "device arena block")) { std::free(b); return nullptr; }
    b->cap = cap;
    b->off = bytes;
    b->next = *list;
    *list = b;
    return b->p;
}

static void arena_reset(ArenaBlock *list) {
    for (; list; list = list->next) list->off = 0;
}

static void arena_free(ArenaBlock **list, int pinned) {
    for (ArenaBlock *b = *list; b;) {
        ArenaBlock *n = b->next;
        if (pinned) cudaFreeHost(b->p); else cudaFree(b->p);
        std::free(b);
        b = n;
    }
    *list = nullptr;
}

static int reserve_bytes(void **ptr, size_t *cap, size_t bytes, const char *what) {
    if (*cap >= bytes) return 1;
    if (*ptr) cudaFree(*ptr);
    *ptr = nullptr;
    *cap = 0;
    if (!cuda_ok(cudaMalloc(ptr, bytes), what)) return 0;
    *cap = bytes;
    return 1;
}

/* ---------------- kernels ---------------- */

__device__ static inline float warp_sum(float v) {
    for (int off = 16; off; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

/* Scalar dequant, identical semantics to the CPU dequant-on-use path. */
__device__ static float weight_at(const void *weights, int fmt, size_t row, int i) {
    const uint8_t *base = static_cast<const uint8_t *>(weights) + row;
    if (fmt == 0) return reinterpret_cast<const float *>(base)[i];
    if (fmt == 1) return static_cast<float>(reinterpret_cast<const int8_t *>(base)[i]);
    const uint8_t *q = base;
    if (fmt == 2) {
        uint8_t v = q[i >> 1];
        return static_cast<float>(((i & 1) ? (v >> 4) : (v & 15)) - 8);
    }
    uint8_t v = q[i >> 2];
    return static_cast<float>(((v >> ((i & 3) * 2)) & 3) - 2);
}

/* One warp per output row, one position per grid.y — decode / tiny batches. */
__global__ static void k_matvec(float *y, const float *x, const void *w, const float *scales,
                                int fmt, int I, int O, size_t rb, int vec) {
    int lane = threadIdx.x & 31;
    int o = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (o >= O) return;
    const float *xs = x + (size_t)blockIdx.y * I;
    const uint8_t *row = (const uint8_t *)w + (size_t)o * rb;
    float sum = 0.f;
    if (!vec) {
        for (int i = lane; i < I; i += 32) sum += xs[i] * weight_at(w, fmt, (size_t)o * rb, i);
    } else if (fmt == 0) {
        const float4 *w4 = (const float4 *)row;
        for (int k = lane; k < (I >> 2); k += 32) {
            float4 a = w4[k];
            const float *xb = xs + ((size_t)k << 2);
            sum += a.x * xb[0] + a.y * xb[1] + a.z * xb[2] + a.w * xb[3];
        }
    } else if (fmt == 1) {
        const uint32_t *wq = (const uint32_t *)row;
        for (int k = lane; k < (I >> 2); k += 32) {
            uint32_t c = wq[k];
            const float *xb = xs + ((size_t)k << 2);
            #pragma unroll
            for (int j = 0; j < 4; j++) sum += xb[j] * (float)(int8_t)(c >> (8 * j));
        }
    } else if (fmt == 2) {
        const uint32_t *wq = (const uint32_t *)row;
        for (int k = lane; k < (I >> 3); k += 32) {
            uint32_t c = wq[k];
            const float *xb = xs + ((size_t)k << 3);
            #pragma unroll
            for (int j = 0; j < 8; j++) sum += xb[j] * (float)((int)((c >> (4 * j)) & 15u) - 8);
        }
    } else {
        const uint32_t *wq = (const uint32_t *)row;
        for (int k = lane; k < (I >> 4); k += 32) {
            uint32_t c = wq[k];
            const float *xb = xs + ((size_t)k << 4);
            #pragma unroll
            for (int j = 0; j < 16; j++) sum += xb[j] * (float)((int)((c >> (2 * j)) & 3u) - 2);
        }
    }
    sum = warp_sum(sum);
    if (!lane) y[(size_t)blockIdx.y * O + o] = sum * (fmt ? scales[o] : 1.0f);
}

/* One output row per blockIdx.x, COLI_TS positions per blockIdx.y: the weight row
 * is fetched once and reused for the whole tile — this is what makes batch work
 * (prefill, MTP verify) compute-bound on the GPU instead of bandwidth-bound. */
#define COLI_TS 8

__global__ static void k_matmul_tile(float *y, const float *x, const void *w, const float *scales,
                                     int fmt, int S, int I, int O, size_t rb, int vec) {
    int o = blockIdx.x, s0 = blockIdx.y * COLI_TS;
    int ns = S - s0; if (ns > COLI_TS) ns = COLI_TS;
    const uint8_t *row = (const uint8_t *)w + (size_t)o * rb;
    float acc[COLI_TS];
    #pragma unroll
    for (int t = 0; t < COLI_TS; t++) acc[t] = 0.f;
    int tid = threadIdx.x, nth = blockDim.x;
    if (!vec) {
        for (int i = tid; i < I; i += nth) {
            float wv = weight_at(w, fmt, (size_t)o * rb, i);
            for (int t = 0; t < ns; t++) acc[t] += wv * x[(size_t)(s0 + t) * I + i];
        }
    } else if (fmt == 0) {
        const float4 *w4 = (const float4 *)row;
        for (int k = tid; k < (I >> 2); k += nth) {
            float4 a = w4[k];
            int ib = k << 2;
            for (int t = 0; t < ns; t++) {
                const float *xb = x + (size_t)(s0 + t) * I + ib;
                acc[t] += a.x * xb[0] + a.y * xb[1] + a.z * xb[2] + a.w * xb[3];
            }
        }
    } else if (fmt == 1) {
        const uint32_t *wq = (const uint32_t *)row;
        for (int k = tid; k < (I >> 2); k += nth) {
            uint32_t c = wq[k];
            int ib = k << 2;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float wv = (float)(int8_t)(c >> (8 * j));
                for (int t = 0; t < ns; t++) acc[t] += wv * x[(size_t)(s0 + t) * I + ib + j];
            }
        }
    } else if (fmt == 2) {
        const uint32_t *wq = (const uint32_t *)row;
        for (int k = tid; k < (I >> 3); k += nth) {
            uint32_t c = wq[k];
            int ib = k << 3;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                float wv = (float)((int)((c >> (4 * j)) & 15u) - 8);
                for (int t = 0; t < ns; t++) acc[t] += wv * x[(size_t)(s0 + t) * I + ib + j];
            }
        }
    } else {
        const uint32_t *wq = (const uint32_t *)row;
        for (int k = tid; k < (I >> 4); k += nth) {
            uint32_t c = wq[k];
            int ib = k << 4;
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                float wv = (float)((int)((c >> (2 * j)) & 3u) - 2);
                for (int t = 0; t < ns; t++) acc[t] += wv * x[(size_t)(s0 + t) * I + ib + j];
            }
        }
    }
    __shared__ float red[COLI_TS][4];
    int lane = tid & 31, wid = tid >> 5, nw = nth >> 5;
    #pragma unroll
    for (int t = 0; t < COLI_TS; t++) {
        float v = warp_sum(acc[t]);
        if (!lane) red[t][wid] = v;
    }
    __syncthreads();
    if (!wid) {
        float sc = fmt ? scales[o] : 1.0f;
        for (int t = lane; t < ns; t += 32) {
            float v = 0;
            for (int z = 0; z < nw; z++) v += red[t][z];
            y[(size_t)(s0 + t) * O + o] = v * sc;
        }
    }
}

__global__ static void k_silu_mul(float *g, const float *u, size_t n) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        float v = g[i];
        g[i] = v / (1.f + expf(-v)) * u[i];
    }
}

static int vec_ok(int fmt, int I) {
    if (fmt == 0 || fmt == 1) return (I & 3) == 0;
    if (fmt == 2) return (I & 7) == 0;
    return (I & 15) == 0;
}

static void launch_matmul(cudaStream_t st, float *y, const float *x, const void *w,
                          const float *scales, int fmt, int S, int I, int O) {
    size_t rb = row_bytes(fmt, I);
    int vec = vec_ok(fmt, I);
    if (S < 4) {
        dim3 grid((unsigned)((O + 3) / 4), (unsigned)S);
        k_matvec<<<grid, 128, 0, st>>>(y, x, w, scales, fmt, I, O, rb, vec);
    } else {
        dim3 grid((unsigned)O, (unsigned)((S + COLI_TS - 1) / COLI_TS));
        k_matmul_tile<<<grid, 128, 0, st>>>(y, x, w, scales, fmt, S, I, O, rb, vec);
    }
}

/* ---------------- public API ---------------- */

extern "C" int coli_cuda_init(const int *devices, int count) {
    int available = 0;
    if (!devices || count < 1 || count > COLI_CUDA_MAX_DEVICES) return 0;
    if (!cuda_ok(cudaGetDeviceCount(&available), "device discovery")) return 0;
    g_nctx = 0;
    for (int i = 0; i < count; i++) {
        int device = devices[i];
        if (device < 0 || device >= available) {
            std::fprintf(stderr, "[CUDA] invalid device %d (available: 0..%d)\n", device, available - 1);
            g_nctx = 0;
            return 0;
        }
        if (find_ctx(device)) {
            std::fprintf(stderr, "[CUDA] duplicate device %d\n", device);
            g_nctx = 0;
            return 0;
        }
        DeviceContext *ctx = &g_ctx[g_nctx];
        *ctx = {};
        ctx->device = device;
        if (!select_ctx(ctx)) { g_nctx = 0; return 0; }
        cudaDeviceProp prop{};
        if (!cuda_ok(cudaGetDeviceProperties(&prop, device), "device properties")) { g_nctx = 0; return 0; }
        if (!cuda_ok(cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking), "stream create")) {
            g_nctx = 0;
            return 0;
        }
        g_nctx++;
        std::fprintf(stderr, "[CUDA] device %d: %s, %.1f GB VRAM, sm_%d%d\n",
                     device, prop.name, prop.totalGlobalMem / 1e9, prop.major, prop.minor);
    }
    return 1;
}

extern "C" void coli_cuda_shutdown(void) {
    for (int i = 0; i < g_nctx; i++) {
        DeviceContext *ctx = &g_ctx[i];
        if (!select_ctx(ctx)) continue;
        cudaStreamSynchronize(ctx->stream);
        cudaStreamDestroy(ctx->stream);
        arena_free(&ctx->pin, 1);
        arena_free(&ctx->dev, 0);
        if (ctx->wbuf) cudaFree(ctx->wbuf);
        if (ctx->wsc) cudaFree(ctx->wsc);
        if (ctx->xbuf) cudaFree(ctx->xbuf);
        if (ctx->ybuf) cudaFree(ctx->ybuf);
        *ctx = {};
    }
    g_nctx = 0;
}

extern "C" int coli_cuda_device_count(void) { return g_nctx; }

extern "C" int coli_cuda_device_at(int index) {
    return index >= 0 && index < g_nctx ? g_ctx[index].device : -1;
}

extern "C" int coli_cuda_mem_info(int device, size_t *free_bytes, size_t *total_bytes) {
    DeviceContext *ctx = find_ctx(device);
    if (!free_bytes || !total_bytes || !select_ctx(ctx)) return 0;
    return cuda_ok(cudaMemGetInfo(free_bytes, total_bytes), "memory info");
}

extern "C" void coli_cuda_stats(int device, size_t *tensor_count, size_t *tensor_bytes) {
    size_t count = 0, bytes = 0;
    for (int i = 0; i < g_nctx; i++) if (device < 0 || g_ctx[i].device == device) {
        count += g_ctx[i].tensor_count;
        bytes += g_ctx[i].tensor_bytes;
    }
    if (tensor_count) *tensor_count = count;
    if (tensor_bytes) *tensor_bytes = bytes;
}

extern "C" int coli_cuda_tensor_upload(ColiCudaTensor **tensor,
                                        const void *weights, const float *scales,
                                        int fmt, int I, int O, int device) {
    DeviceContext *ctx = find_ctx(device);
    if (!tensor || !weights || I < 1 || O < 1 || !select_ctx(ctx)) return 0;
    size_t rb = row_bytes(fmt, I);
    if (!rb || (fmt && !scales)) return 0;
    if (*tensor) {
        ColiCudaTensor *t = *tensor;
        return t->fmt == fmt && t->I == I && t->O == O && t->device == device;
    }
    ColiCudaTensor *t = static_cast<ColiCudaTensor *>(std::calloc(1, sizeof(*t)));
    if (!t) return 0;
    t->fmt = fmt; t->I = I; t->O = O; t->device = device; t->weight_bytes = rb * (size_t)O;
    if (!cuda_ok(cudaMalloc(&t->weights, t->weight_bytes), "tensor allocation") ||
        !cuda_ok(cudaMemcpy(t->weights, weights, t->weight_bytes, cudaMemcpyHostToDevice), "tensor upload")) {
        coli_cuda_tensor_free(t);
        return 0;
    }
    if (fmt) {
        if (!cuda_ok(cudaMalloc(&t->scales, (size_t)O * sizeof(float)), "scale allocation") ||
            !cuda_ok(cudaMemcpy(t->scales, scales, (size_t)O * sizeof(float), cudaMemcpyHostToDevice), "scale upload")) {
            coli_cuda_tensor_free(t);
            return 0;
        }
    }
    t->tracked = 1;
    ctx->tensor_count++;
    ctx->tensor_bytes += t->weight_bytes + (fmt ? (size_t)O * sizeof(float) : 0);
    *tensor = t;
    return 1;
}

extern "C" int coli_cuda_matmul(ColiCudaTensor **tensor,
                                 float *y, const float *x,
                                 const void *weights, const float *scales,
                                 int fmt, int S, int I, int O, int device) {
    if (S < 1 || !coli_cuda_tensor_upload(tensor, weights, scales, fmt, I, O, device)) return 0;
    ColiCudaTensor *t = *tensor;
    DeviceContext *ctx = find_ctx(t->device);
    if (!select_ctx(ctx)) return 0;
    size_t xb = (size_t)S * I * sizeof(float), yb = (size_t)S * O * sizeof(float);
    if (!reserve_bytes((void **)&ctx->xbuf, &ctx->x_cap, xb, "x scratch") ||
        !reserve_bytes((void **)&ctx->ybuf, &ctx->y_cap, yb, "y scratch")) return 0;
    if (!cuda_ok(cudaMemcpyAsync(ctx->xbuf, x, xb, cudaMemcpyHostToDevice, ctx->stream), "input upload")) return 0;
    launch_matmul(ctx->stream, ctx->ybuf, ctx->xbuf, t->weights, t->scales, fmt, S, I, O);
    if (!cuda_ok(cudaGetLastError(), "matmul launch") ||
        !cuda_ok(cudaMemcpyAsync(y, ctx->ybuf, yb, cudaMemcpyDeviceToHost, ctx->stream), "output download") ||
        !cuda_ok(cudaStreamSynchronize(ctx->stream), "matmul sync")) return 0;
    return 1;
}

extern "C" int coli_cuda_matmul_stream(float *y, const float *x,
                                        const void *weights, const float *scales,
                                        int fmt, int S, int I, int O, int device) {
    DeviceContext *ctx = find_ctx(device);
    if (S < 1 || !weights || (fmt && !scales) || !select_ctx(ctx)) return 0;
    size_t rb = row_bytes(fmt, I);
    if (!rb) return 0;
    size_t wb = rb * (size_t)O;
    size_t xb = (size_t)S * I * sizeof(float), yb = (size_t)S * O * sizeof(float);
    if (!reserve_bytes(&ctx->wbuf, &ctx->wcap, wb, "streamed weights scratch") ||
        !reserve_bytes((void **)&ctx->xbuf, &ctx->x_cap, xb, "x scratch") ||
        !reserve_bytes((void **)&ctx->ybuf, &ctx->y_cap, yb, "y scratch")) return 0;
    if (fmt && !reserve_bytes((void **)&ctx->wsc, &ctx->wsc_cap, (size_t)O * sizeof(float),
                              "streamed scales scratch")) return 0;
    if (!cuda_ok(cudaMemcpyAsync(ctx->wbuf, weights, wb, cudaMemcpyHostToDevice, ctx->stream), "weights upload")) return 0;
    if (fmt && !cuda_ok(cudaMemcpyAsync(ctx->wsc, scales, (size_t)O * sizeof(float),
                                        cudaMemcpyHostToDevice, ctx->stream), "scales upload")) return 0;
    if (!cuda_ok(cudaMemcpyAsync(ctx->xbuf, x, xb, cudaMemcpyHostToDevice, ctx->stream), "input upload")) return 0;
    launch_matmul(ctx->stream, ctx->ybuf, ctx->xbuf, ctx->wbuf, fmt ? ctx->wsc : nullptr, fmt, S, I, O);
    if (!cuda_ok(cudaGetLastError(), "streamed matmul launch") ||
        !cuda_ok(cudaMemcpyAsync(y, ctx->ybuf, yb, cudaMemcpyDeviceToHost, ctx->stream), "output download") ||
        !cuda_ok(cudaStreamSynchronize(ctx->stream), "streamed matmul sync")) return 0;
    return 1;
}

extern "C" int coli_cuda_ffn_enqueue(ColiCudaTensor *g, ColiCudaTensor *u, ColiCudaTensor *d,
                                      const float *x, const int *rows, int nr, float **hh_out) {
    if (!g || !u || !d || !x || !rows || nr < 1 || !hh_out) return 0;
    if (g->device != u->device || g->device != d->device) return 0;
    if (u->I != g->I || u->O != g->O || d->I != g->O || d->O != g->I) return 0;
    DeviceContext *ctx = find_ctx(g->device);
    if (!select_ctx(ctx)) return 0;
    int D = g->I, I = g->O;
    size_t xb = (size_t)nr * D * sizeof(float), ib = (size_t)nr * I * sizeof(float);
    float *px = (float *)arena_take(&ctx->pin, xb, 1);
    float *phh = (float *)arena_take(&ctx->pin, xb, 1);
    float *dx = (float *)arena_take(&ctx->dev, xb, 0);
    float *dgg = (float *)arena_take(&ctx->dev, ib, 0);
    float *duu = (float *)arena_take(&ctx->dev, ib, 0);
    float *dhh = (float *)arena_take(&ctx->dev, xb, 0);
    if (!px || !phh || !dx || !dgg || !duu || !dhh) return 0;
    for (int r = 0; r < nr; r++)
        std::memcpy(px + (size_t)r * D, x + (size_t)rows[r] * D, (size_t)D * sizeof(float));
    if (!cuda_ok(cudaMemcpyAsync(dx, px, xb, cudaMemcpyHostToDevice, ctx->stream), "ffn input upload")) return 0;
    launch_matmul(ctx->stream, dgg, dx, g->weights, g->scales, g->fmt, nr, D, I);
    launch_matmul(ctx->stream, duu, dx, u->weights, u->scales, u->fmt, nr, D, I);
    size_t n = (size_t)nr * I;
    unsigned blocks = (unsigned)((n + 255) / 256);
    if (blocks > 4096) blocks = 4096;
    k_silu_mul<<<blocks, 256, 0, ctx->stream>>>(dgg, duu, n);
    launch_matmul(ctx->stream, dhh, dgg, d->weights, d->scales, d->fmt, nr, I, D);
    if (!cuda_ok(cudaGetLastError(), "ffn launch") ||
        !cuda_ok(cudaMemcpyAsync(phh, dhh, xb, cudaMemcpyDeviceToHost, ctx->stream), "ffn output download"))
        return 0;
    *hh_out = phh;
    return 1;
}

extern "C" int coli_cuda_sync(int device) {
    int ok = 1;
    for (int i = 0; i < g_nctx; i++) {
        DeviceContext *ctx = &g_ctx[i];
        if (device >= 0 && ctx->device != device) continue;
        if (!select_ctx(ctx) || !cuda_ok(cudaStreamSynchronize(ctx->stream), "stream sync")) { ok = 0; continue; }
        arena_reset(ctx->pin);
        arena_reset(ctx->dev);
    }
    return ok;
}

extern "C" void coli_cuda_tensor_free(ColiCudaTensor *tensor) {
    if (!tensor) return;
    DeviceContext *ctx = find_ctx(tensor->device);
    if (ctx) select_ctx(ctx);
    if (tensor->tracked && ctx) {
        size_t bytes = tensor->weight_bytes + (tensor->fmt ? (size_t)tensor->O * sizeof(float) : 0);
        if (ctx->tensor_count) ctx->tensor_count--;
        if (ctx->tensor_bytes >= bytes) ctx->tensor_bytes -= bytes;
    }
    if (tensor->weights) cudaFree(tensor->weights);
    if (tensor->scales) cudaFree(tensor->scales);
    std::free(tensor);
}

extern "C" size_t coli_cuda_tensor_bytes(const ColiCudaTensor *tensor) {
    return tensor ? tensor->weight_bytes + (tensor->fmt ? (size_t)tensor->O * sizeof(float) : 0) : 0;
}

extern "C" int coli_cuda_tensor_device(const ColiCudaTensor *tensor) {
    return tensor ? tensor->device : -1;
}
