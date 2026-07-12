#include "../backend_cuda.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

static int close_enough(const float *got, const float *want, int n) {
    for (int i = 0; i < n; i++) {
        if (std::fabs(got[i] - want[i]) > 1e-3f) {
            std::fprintf(stderr, "mismatch %d: got %.6f want %.6f\n", i, got[i], want[i]);
            return 0;
        }
    }
    return 1;
}

/* CPU reference: dequant-on-use, identical container semantics to the engine. */
static float wref(const void *w, const float *s, int fmt, int I, int o, int i) {
    if (fmt == 0) return ((const float *)w)[(size_t)o * I + i];
    if (fmt == 1) return ((const int8_t *)w)[(size_t)o * I + i] * s[o];
    if (fmt == 2) {
        const uint8_t *q = (const uint8_t *)w + (size_t)o * ((I + 1) / 2);
        uint8_t v = q[i >> 1];
        return (float)(((i & 1) ? (v >> 4) : (v & 15)) - 8) * s[o];
    }
    const uint8_t *q = (const uint8_t *)w + (size_t)o * ((I + 3) / 4);
    uint8_t v = q[i >> 2];
    return (float)(((v >> ((i & 3) * 2)) & 3) - 2) * s[o];
}

static void ref_matmul(const float *x, const void *w, const float *s,
                       int fmt, int S, int I, int O, float *y) {
    for (int ss = 0; ss < S; ss++)
        for (int o = 0; o < O; o++) {
            float a = 0;
            for (int i = 0; i < I; i++) a += x[(size_t)ss * I + i] * wref(w, s, fmt, I, o, i);
            y[(size_t)ss * O + o] = a;
        }
}

static uint32_t g_seed = 1;
static uint32_t rnd(void) { return g_seed = g_seed * 1664525u + 1013904223u; }
/* exact binary fractions: sums stay exactly representable, so any summation
 * order (warp reduction vs serial) produces identical f32 results */
static float frac8(void) { return (float)((int)(rnd() % 17) - 8) / 8.0f; }

/* ---- batch matmul (tile kernel) + streamed matmul over every format ---- */
static int test_batch_and_stream(int device) {
    enum { S = 13, I = 16, O = 5 };
    static float x[S * I], y[S * O], want[S * O];
    for (int i = 0; i < S * I; i++) x[i] = frac8();

    static int8_t q8[O * I];
    static uint8_t q4[O * (I / 2)], q2[O * (I / 4)];
    static float w32[O * I], sc[O];
    for (int i = 0; i < O * I; i++) q8[i] = (int8_t)((int)(rnd() % 15) - 7);
    for (int i = 0; i < O * (I / 2); i++) q4[i] = (uint8_t)(rnd() & 0xff);
    for (int i = 0; i < O * (I / 4); i++) q2[i] = (uint8_t)(rnd() & 0xff);
    for (int i = 0; i < O * I; i++) w32[i] = frac8();
    for (int o = 0; o < O; o++) sc[o] = 1.0f / (float)(1 + (o & 3));

    struct { const void *w; const float *s; int fmt; } cases[4] = {
        {w32, nullptr, 0}, {q8, sc, 1}, {q4, sc, 2}, {q2, sc, 3},
    };
    for (int c = 0; c < 4; c++) {
        ref_matmul(x, cases[c].w, cases[c].s, cases[c].fmt, S, I, O, want);
        /* resident tensor path (S>=4 exercises the tile kernel) */
        ColiCudaTensor *t = nullptr;
        if (!coli_cuda_matmul(&t, y, x, cases[c].w, cases[c].s, cases[c].fmt, S, I, O, device) ||
            !close_enough(y, want, S * O)) {
            std::fprintf(stderr, "batch matmul fmt=%d failed\n", cases[c].fmt);
            return 0;
        }
        /* small S exercises the warp-per-row kernel on the same tensor */
        if (!coli_cuda_matmul(&t, y, x, cases[c].w, cases[c].s, cases[c].fmt, 2, I, O, device) ||
            !close_enough(y, want, 2 * O)) {
            std::fprintf(stderr, "matvec fmt=%d failed\n", cases[c].fmt);
            return 0;
        }
        coli_cuda_tensor_free(t);
        /* streamed path: no persistent tensor, weights cross PCIe per call */
        std::memset(y, 0, sizeof(y));
        if (!coli_cuda_matmul_stream(y, x, cases[c].w, cases[c].s, cases[c].fmt, S, I, O, device) ||
            !close_enough(y, want, S * O)) {
            std::fprintf(stderr, "streamed matmul fmt=%d failed\n", cases[c].fmt);
            return 0;
        }
    }

    /* scalar fallback: I=6 breaks every vector alignment rule */
    {
        enum { So = 5, Io = 6, Oo = 3 };
        static float xo[So * Io], yo[So * Oo];
        static int8_t qo[Oo * Io];
        static float so[Oo], wanto[So * Oo];
        for (int i = 0; i < So * Io; i++) xo[i] = frac8();
        for (int i = 0; i < Oo * Io; i++) qo[i] = (int8_t)((int)(rnd() % 15) - 7);
        for (int o = 0; o < Oo; o++) so[o] = 0.5f;
        ref_matmul(xo, qo, so, 1, So, Io, Oo, wanto);
        if (!coli_cuda_matmul_stream(yo, xo, qo, so, 1, So, Io, Oo, device) ||
            !close_enough(yo, wanto, So * Oo)) {
            std::fprintf(stderr, "scalar-fallback matmul failed\n");
            return 0;
        }
    }
    return 1;
}

/* ---- async expert FFN: gather, enqueue x2, sync, collect ---- */
static int test_ffn_async(int device) {
    enum { D = 16, FI = 8, NPOS = 6, NR = 3 };
    static float x[NPOS * D];
    for (int i = 0; i < NPOS * D; i++) x[i] = frac8();
    static const int rows[NR] = {0, 4, 2};   /* out of order on purpose */

    static int8_t gq[2][FI * D], uq[2][FI * D], dq[2][D * FI];
    static float gs[2][FI], us[2][FI], ds[2][D];
    ColiCudaTensor *gt[2] = {nullptr, nullptr}, *ut[2] = {nullptr, nullptr}, *dt[2] = {nullptr, nullptr};
    for (int e = 0; e < 2; e++) {
        for (int i = 0; i < FI * D; i++) gq[e][i] = (int8_t)((int)(rnd() % 15) - 7);
        for (int i = 0; i < FI * D; i++) uq[e][i] = (int8_t)((int)(rnd() % 15) - 7);
        for (int i = 0; i < D * FI; i++) dq[e][i] = (int8_t)((int)(rnd() % 15) - 7);
        for (int i = 0; i < FI; i++) { gs[e][i] = 0.25f; us[e][i] = 0.5f; }
        for (int i = 0; i < D; i++) ds[e][i] = 0.125f;
        if (!coli_cuda_tensor_upload(&gt[e], gq[e], gs[e], 1, D, FI, device) ||
            !coli_cuda_tensor_upload(&ut[e], uq[e], us[e], 1, D, FI, device) ||
            !coli_cuda_tensor_upload(&dt[e], dq[e], ds[e], 1, FI, D, device)) {
            std::fprintf(stderr, "ffn tensor upload failed\n");
            return 0;
        }
    }

    /* dimension mismatch must be rejected before touching the stream */
    {
        float *bad = nullptr;
        if (coli_cuda_ffn_enqueue(gt[0], ut[0], gt[0], x, rows, NR, &bad)) {
            std::fprintf(stderr, "ffn accepted mismatched down projection\n");
            return 0;
        }
    }

    for (int round = 0; round < 2; round++) {   /* round 2 reuses the reset arenas */
        float *hh[2] = {nullptr, nullptr};
        for (int e = 0; e < 2; e++)
            if (!coli_cuda_ffn_enqueue(gt[e], ut[e], dt[e], x, rows, NR, &hh[e])) {
                std::fprintf(stderr, "ffn enqueue %d failed (round %d)\n", e, round);
                return 0;
            }
        if (!coli_cuda_sync(-1)) return 0;
        for (int e = 0; e < 2; e++) {
            float want[NR * D];
            for (int r = 0; r < NR; r++) {
                const float *xr = x + (size_t)rows[r] * D;
                float gv[FI], uv[FI];
                for (int o = 0; o < FI; o++) {
                    float a = 0, b = 0;
                    for (int i = 0; i < D; i++) {
                        a += xr[i] * wref(gq[e], gs[e], 1, D, o, i);
                        b += xr[i] * wref(uq[e], us[e], 1, D, o, i);
                    }
                    gv[o] = a / (1.f + std::exp(-a)) * b;
                    (void)uv;
                }
                for (int o = 0; o < D; o++) {
                    float a = 0;
                    for (int i = 0; i < FI; i++) a += gv[i] * wref(dq[e], ds[e], 1, FI, o, i);
                    want[(size_t)r * D + o] = a;
                }
            }
            if (!close_enough(hh[e], want, NR * D)) {
                std::fprintf(stderr, "ffn result mismatch expert %d round %d\n", e, round);
                return 0;
            }
        }
    }
    for (int e = 0; e < 2; e++) {
        coli_cuda_tensor_free(gt[e]);
        coli_cuda_tensor_free(ut[e]);
        coli_cuda_tensor_free(dt[e]);
    }
    return 1;
}

int main(int argc, char **argv) {
    int devices[COLI_CUDA_MAX_DEVICES], ndev = argc > 1 ? argc - 1 : 1;
    if (ndev > COLI_CUDA_MAX_DEVICES) return 2;
    for (int i = 0; i < ndev; i++) devices[i] = argc > 1 ? std::atoi(argv[i + 1]) : 0;
    if (!coli_cuda_init(devices, ndev)) return 77;
    if (coli_cuda_device_count() != ndev) return 1;
    int d0 = devices[0], d1 = devices[ndev > 1 ? 1 : 0];
    size_t count = 99, bytes = 99;
    coli_cuda_stats(-1, &count, &bytes);
    if (count || bytes) return 1;
    const float x[8] = {1, -2, 3, -4, 2, 1, -1, 0.5f};
    float got[4];

    const int8_t q8[8] = {1, 2, 3, 4, -1, 2, -3, 4};
    const float s8[2] = {0.5f, 2.0f};
    const float want8[4] = {-5.0f, -60.0f, 1.5f, 10.0f};
    ColiCudaTensor *t8 = nullptr;
    if (!coli_cuda_tensor_upload(&t8, q8, s8, 1, 4, 2, d0)) return 1;
    if (coli_cuda_tensor_upload(&t8, q8, s8, 1, 5, 2, d0)) return 1;
    if (ndev > 1 && coli_cuda_tensor_upload(&t8, q8, s8, 1, 4, 2, d1)) return 1;
    if (!coli_cuda_matmul(&t8, got, x, q8, s8, 1, 2, 4, 2, d0) || !close_enough(got, want8, 4)) return 1;

    /* Rows [-8,-1,0,7] and [1,2,3,4], packed low nibble first. */
    const uint8_t q4[4] = {0x70, 0xf8, 0xa9, 0xcb};
    const float s4[2] = {1.0f, 0.25f};
    const float want4[2] = {-34.0f, -2.5f};
    ColiCudaTensor *t4 = nullptr;
    if (!coli_cuda_matmul(&t4, got, x, q4, s4, 2, 1, 4, 2, d1) || !close_enough(got, want4, 2)) return 1;

    const uint8_t q2[2] = {0xe4, 0x1b};
    const float s2[2] = {0.5f, 2.0f};
    const float want2[2] = {-2.0f, 12.0f};
    ColiCudaTensor *t2 = nullptr;
    if (!coli_cuda_matmul(&t2, got, x, q2, s2, 3, 1, 4, 2, d1) || !close_enough(got, want2, 2)) return 1;

    const float wf[8] = {1, 0, -1, 2, 0.5f, 0.5f, 0.5f, 0.5f};
    const float wantf[2] = {-10.0f, -1.0f};
    ColiCudaTensor *tf = nullptr;
    if (!coli_cuda_matmul(&tf, got, x, wf, nullptr, 0, 1, 4, 2, d0) || !close_enough(got, wantf, 2)) return 1;

    coli_cuda_stats(-1, &count, &bytes);
    if (count != 4 || bytes != 70) {
        std::fprintf(stderr, "unexpected CUDA stats: %zu tensors, %zu bytes\n", count, bytes);
        return 1;
    }
    if (coli_cuda_tensor_device(t8) != d0 || coli_cuda_tensor_device(tf) != d0 ||
        coli_cuda_tensor_device(t4) != d1 || coli_cuda_tensor_device(t2) != d1) return 1;
    coli_cuda_stats(d0, &count, &bytes);
    if (ndev > 1) {
        if (count != 2 || bytes != 48) return 1;
        coli_cuda_stats(d1, &count, &bytes);
        if (count != 2 || bytes != 22) return 1;
    } else if (count != 4 || bytes != 70) return 1;

    coli_cuda_tensor_free(t8);
    coli_cuda_tensor_free(t4);
    coli_cuda_tensor_free(t2);
    coli_cuda_tensor_free(tf);
    coli_cuda_stats(-1, &count, &bytes);
    if (count || bytes) return 1;

    /* stage-2 paths: tile kernel, streamed weights, async expert FFN */
    if (!coli_cuda_sync(-1)) return 1;                    /* sync with nothing pending is a no-op */
    if (!test_batch_and_stream(d0)) return 1;
    if (ndev > 1 && !test_batch_and_stream(d1)) return 1;
    if (!test_ffn_async(d0)) return 1;
    coli_cuda_stats(-1, &count, &bytes);
    if (count || bytes) {
        std::fprintf(stderr, "leak: %zu tensors, %zu bytes still tracked\n", count, bytes);
        return 1;
    }

    coli_cuda_shutdown();
    std::printf("cuda backend: q8/q4/q2/f32 matvec+tile+stream+ffn ok on %d device(s)\n", ndev);
    return 0;
}
