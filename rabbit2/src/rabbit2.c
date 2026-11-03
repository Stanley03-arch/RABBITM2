/*
 * RABBIT2 -- milestone 2: MoE transformer engine that can load real models (e.g. OLMoE)
 *
 * Build:  gcc -O2 -std=c99 -o build/rabbit2 src/rabbit2.c -lm        (MSVC: cl /O2 src\rabbit2.c)
 *
 * Usage:
 *   rabbit2 <model.rbt2> --dump <logits.bin> <routing.bin> [--ctx N] <token> ...
 *       write float32 logits [n_tokens, vocab] and int32 expert choices [n_tokens, n_layers, top_k]
 *   rabbit2 <model.rbt2> --gen <n> [--eos id] [--ctx N] <token> ...
 *       greedy-generate up to n tokens after the prompt; one token id per line on stdout
 *   rabbit2 <model.rbt2> --info
 *       print the model config and byte layout (used by tests)
 *   --ctx N caps the context (KV cache) length; default min(max_seq, 512)
 *   --expert-cache-mib N enables a bounded local expert-buffer cache (experimental)
 *   --kernel scalar|auto|avx2 selects the experimental CPU matvec kernel
 *
 * What changed from M1:
 *   - weights may be float32 or bfloat16 (vectors/norms are always float32)
 *   - the file is memory-mapped, so a model larger than RAM works (the OS pages experts in)
 *   - QK-norm and optional top-k renormalisation (OLMoE: qk_norm=1, norm_topk=0)
 *   - greedy generation, context cap, timing on stderr
 *
 * ALL expert weight access goes through get_expert(): mmap by default, or an optional
 * bounded synchronous local-file cache. Remote streaming and strict total-RAM limits are future work.
 */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* AVX2 is compiled only with GCC/Clang on x86 and selected at runtime.
 * Other compilers/architectures retain the portable scalar implementation. */
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#define RBT_HAVE_AVX2_KERNEL 1
#include <immintrin.h>
#else
#define RBT_HAVE_AVX2_KERNEL 0
#endif

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#endif

#define ALIGN 4096  /* header and expert blobs */
#define TALIGN 64   /* every other tensor */
#define HEADER_BYTES 68 /* 4 magic + 4 version + 13*int32 + 2*float32 */
#define MAX_TOPK 64
#define DT_F32 0
#define DT_BF16 1

typedef struct {
    int32_t vocab, dim, n_layers, n_heads, n_kv_heads, head_dim;
    int32_t n_experts, top_k, expert_hidden, max_seq;
    int32_t dtype, qk_norm, norm_topk;
    float rope_theta, rms_eps;
} Config;
typedef char config_size_check[sizeof(Config) == 60 ? 1 : -1]; /* must match the on-disk header */

typedef struct {
    size_t attn_norm, wq, wk, wv, wo, q_norm, k_norm, ffn_norm, router, experts;
} LayerOff;

typedef struct {
    unsigned char *base;
    size_t size;
#ifdef _WIN32
    HANDLE hfile, hmap;
#endif
} Mapped;

typedef struct {
    int layer, expert, valid;
    uint64_t last_used;
    unsigned char *data;
} ExpertSlot;

typedef struct {
    FILE *file;
    ExpertSlot *slots;
    size_t capacity, resident;
    uint64_t clock, hits, misses, evictions, bytes_read;
} ExpertCache;

typedef struct {
    Config c;
    size_t tok_emb, final_norm, lm_head, expert_stride;
    size_t esz; /* bytes per matrix element: 4 (f32) or 2 (bf16) */
    LayerOff *layers;
    Mapped map;
    ExpertCache cache;
} Model;

typedef struct {
    int ctx;
    float *x, *xb, *q, *k, *v, *att, *attn_out, *router, *h1, *h3, *tmp, *moe, *logits;
    float *kcache, *vcache; /* [n_layers][ctx][kv_dim] */
    unsigned char *used;
} State;

static size_t align_to(size_t x, size_t a) { return (x + a - 1) / a * a; }

static double now_sec(void) {
#ifdef _WIN32
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
#endif
}

/* ------------------------------------------------------------------ loading */

static int map_file(Mapped *mp, const char *path) {
#ifdef _WIN32
    HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE) return 0;
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(h, &sz)) { CloseHandle(h); return 0; }
    HANDLE hm = CreateFileMappingA(h, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hm) { CloseHandle(h); return 0; }
    void *p = MapViewOfFile(hm, FILE_MAP_READ, 0, 0, 0);
    if (!p) { CloseHandle(hm); CloseHandle(h); return 0; }
    mp->base = (unsigned char *)p;
    mp->size = (size_t)sz.QuadPart; /* build 64-bit: a 32-bit process cannot map a multi-GB model */
    mp->hfile = h;
    mp->hmap = hm;
    return 1;
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return 0; }
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return 0;
    mp->base = (unsigned char *)p;
    mp->size = (size_t)st.st_size;
    return 1;
#endif
}

/* Mirrors layout() in ref/rbt2.py exactly. */
static size_t take(size_t *off, size_t nbytes) {
    *off = align_to(*off, TALIGN);
    size_t start = *off;
    *off += nbytes;
    return start;
}

static void layout(Model *m) {
    const Config *c = &m->c;
    const size_t E = m->esz, F = sizeof(float);
    const size_t q_dim = (size_t)c->n_heads * c->head_dim, kv_dim = (size_t)c->n_kv_heads * c->head_dim;
    size_t off = ALIGN;

    m->tok_emb = take(&off, (size_t)c->vocab * c->dim * E);
    m->expert_stride = align_to(3 * (size_t)c->expert_hidden * c->dim * E, ALIGN);
    for (int l = 0; l < c->n_layers; l++) {
        LayerOff *ly = &m->layers[l];
        ly->attn_norm = take(&off, (size_t)c->dim * F);
        ly->wq = take(&off, q_dim * c->dim * E);
        ly->wk = take(&off, kv_dim * c->dim * E);
        ly->wv = take(&off, kv_dim * c->dim * E);
        ly->wo = take(&off, (size_t)c->dim * q_dim * E);
        if (c->qk_norm) {
            ly->q_norm = take(&off, q_dim * F);
            ly->k_norm = take(&off, kv_dim * F);
        }
        ly->ffn_norm = take(&off, (size_t)c->dim * F);
        ly->router = take(&off, (size_t)c->n_experts * c->dim * E);
        off = align_to(off, ALIGN);
        ly->experts = off;
        off += (size_t)c->n_experts * m->expert_stride;
    }
    m->final_norm = take(&off, (size_t)c->dim * F);
    m->lm_head = take(&off, (size_t)c->vocab * c->dim * E);
}

static size_t layout_end(const Model *m) {
    return m->lm_head + (size_t)m->c.vocab * m->c.dim * m->esz;
}

static int load_model(Model *m, const char *path) {
    if (!map_file(&m->map, path)) { fprintf(stderr, "cannot open/map %s\n", path); return 0; }
    const unsigned char *d = m->map.base;
    if (m->map.size < HEADER_BYTES || memcmp(d, "RBT2", 4) != 0) {
        fprintf(stderr, "bad magic: not a .rbt2 file\n");
        return 0;
    }
    uint32_t version;
    memcpy(&version, d + 4, 4);
    if (version != 2) { fprintf(stderr, "unsupported .rbt2 version %u (need 2)\n", version); return 0; }
    memcpy(&m->c, d + 8, sizeof(Config));

    const Config *c = &m->c;
    if (c->vocab <= 0 || c->dim <= 0 || c->n_layers <= 0 || c->n_heads <= 0 || c->n_kv_heads <= 0 ||
        c->head_dim <= 0 || c->n_experts <= 0 || c->expert_hidden <= 0 || c->max_seq <= 0 ||
        c->top_k <= 0 || c->top_k > c->n_experts || c->top_k > MAX_TOPK || c->head_dim % 2 != 0 ||
        c->n_heads % c->n_kv_heads != 0 || (c->dtype != DT_F32 && c->dtype != DT_BF16)) {
        fprintf(stderr, "invalid config in header\n");
        return 0;
    }
    m->esz = (c->dtype == DT_BF16) ? 2 : 4;
    m->layers = (LayerOff *)calloc((size_t)c->n_layers, sizeof(LayerOff));
    layout(m);
    if (layout_end(m) > m->map.size) {
        fprintf(stderr, "file too small for its own config (%llu < %llu)\n",
                (unsigned long long)m->map.size, (unsigned long long)layout_end(m));
        return 0;
    }
    return 1;
}

static const float *VEC(const Model *m, size_t off) { return (const float *)(m->map.base + off); }
static const void *MAT(const Model *m, size_t off) { return (const void *)(m->map.base + off); }

/* Optional explicit local-file expert cache. The rest of the model remains mmap'd.
 * Budget covers expert payloads only, not metadata, stdio, mmap/page cache, KV or OS.
 * The returned pointer is valid until the next get_expert() call. Single-threaded. */
static int expert_cache_init(Model *m, const char *path, size_t budget) {
    ExpertCache *c = &m->cache;
    if (budget < m->expert_stride) {
        fprintf(stderr, "expert cache budget too small: need at least %llu bytes for one expert\n",
                (unsigned long long)m->expert_stride);
        return 0;
    }
    size_t cap = budget / m->expert_stride;
    size_t total_experts = (size_t)m->c.n_layers * (size_t)m->c.n_experts;
    if (cap > total_experts) cap = total_experts;
    if (cap > SIZE_MAX / sizeof(ExpertSlot)) { fprintf(stderr, "expert cache too large\n"); return 0; }
    c->slots = (ExpertSlot *)calloc(cap, sizeof(ExpertSlot));
    if (!c->slots) { fprintf(stderr, "cannot allocate expert cache metadata\n"); return 0; }
    c->file = fopen(path, "rb");
    if (!c->file) { fprintf(stderr, "cannot open model for expert reads\n"); free(c->slots); c->slots = NULL; return 0; }
    c->capacity = cap;
    return 1;
}

static void expert_cache_close(ExpertCache *c) {
    if (c->file) fclose(c->file);
    for (size_t i = 0; i < c->capacity; i++) free(c->slots[i].data);
    free(c->slots);
}

static const unsigned char *get_expert(Model *m, int layer, int e) {
    size_t offset = m->layers[layer].experts + (size_t)e * m->expert_stride;
    ExpertCache *c = &m->cache;
    if (!c->file) return m->map.base + offset;

    ExpertSlot *victim = NULL;
    for (size_t i = 0; i < c->capacity; i++) {
        ExpertSlot *s = &c->slots[i];
        if (s->valid && s->layer == layer && s->expert == e) {
            s->last_used = ++c->clock;
            c->hits++;
            return s->data;
        }
        if (!victim || (!s->valid && victim->valid) ||
            (s->valid == victim->valid && s->last_used < victim->last_used)) victim = s;
    }
    c->misses++;
    if (!victim->data) {
        victim->data = (unsigned char *)malloc(m->expert_stride);
        if (!victim->data) { fprintf(stderr, "cannot allocate expert cache slot\n"); return NULL; }
        c->resident++;
    }
    if (victim->valid) c->evictions++;
    victim->valid = 0;
    if (offset > INT64_MAX) { fprintf(stderr, "expert offset exceeds 64-bit file limit\n"); return NULL; }
#ifdef _WIN32
    if (_fseeki64(c->file, (__int64)offset, SEEK_SET) != 0) {
#else
    if (fseeko(c->file, (off_t)offset, SEEK_SET) != 0) {
#endif
        fprintf(stderr, "cannot seek to expert weights\n"); return NULL;
    }
    if (fread(victim->data, 1, m->expert_stride, c->file) != m->expert_stride) {
        fprintf(stderr, "cannot read complete expert weights (layer %d expert %d)\n", layer, e);
        return NULL;
    }
    c->bytes_read += m->expert_stride;
    victim->layer = layer; victim->expert = e; victim->last_used = ++c->clock; victim->valid = 1;
    return victim->data;
}

/* ------------------------------------------------------------------ math */

static inline float bf16f(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

/* Portable default; AVX2 is opt-in until real-model and Windows validation. */
static int use_avx2 = 0;

static int cpu_has_avx2(void) {
#if RBT_HAVE_AVX2_KERNEL
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") != 0;
#else
    return 0;
#endif
}

#if RBT_HAVE_AVX2_KERNEL
/* No FMA: preserve the no-fast-math policy. Accumulation order differs from scalar. */
__attribute__((target("avx2,no-fma")))
static void matvec_avx2(float *out, const void *W, int dtype, const float *x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const float *wf = dtype == DT_F32 ? (const float *)W + (size_t)r * cols : NULL;
        const uint16_t *wb = dtype == DT_BF16 ? (const uint16_t *)W + (size_t)r * cols : NULL;
        __m256 a = _mm256_setzero_ps(), b = _mm256_setzero_ps();
        int i = 0;
        for (; i + 15 < cols; i += 16) {
            __m256 w0, w1;
            if (wf) {
                w0 = _mm256_loadu_ps(wf + i);
                w1 = _mm256_loadu_ps(wf + i + 8);
            } else {
                __m128i x0 = _mm_loadu_si128((const __m128i *)(const void *)(wb + i));
                __m128i x1 = _mm_loadu_si128((const __m128i *)(const void *)(wb + i + 8));
                w0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(x0), 16));
                w1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(x1), 16));
            }
            a = _mm256_add_ps(a, _mm256_mul_ps(w0, _mm256_loadu_ps(x + i)));
            b = _mm256_add_ps(b, _mm256_mul_ps(w1, _mm256_loadu_ps(x + i + 8)));
        }
        float lanes[8];
        _mm256_storeu_ps(lanes, _mm256_add_ps(a, b));
        float sum = 0.0f;
        for (int k = 0; k < 8; k++) sum += lanes[k];
        for (; i < cols; i++) sum += (wf ? wf[i] : bf16f(wb[i])) * x[i];
        out[r] = sum;
    }
}
#endif

/* out[r] = dot(W[r,:], x).  W is [rows, cols] row-major in the given dtype. */
static void matvec(float *out, const void *W, int dtype, const float *x, int rows, int cols) {
#if RBT_HAVE_AVX2_KERNEL
    if (use_avx2) { matvec_avx2(out, W, dtype, x, rows, cols); return; }
#endif
    if (dtype == DT_F32) {
        const float *w = (const float *)W;
        for (int r = 0; r < rows; r++, w += cols) {
            float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            int i = 0;
            for (; i + 3 < cols; i += 4) {
                s0 += w[i] * x[i];
                s1 += w[i + 1] * x[i + 1];
                s2 += w[i + 2] * x[i + 2];
                s3 += w[i + 3] * x[i + 3];
            }
            for (; i < cols; i++) s0 += w[i] * x[i];
            out[r] = (s0 + s1) + (s2 + s3);
        }
    } else {
        const uint16_t *w = (const uint16_t *)W;
        for (int r = 0; r < rows; r++, w += cols) {
            float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            int i = 0;
            for (; i + 3 < cols; i += 4) {
                s0 += bf16f(w[i]) * x[i];
                s1 += bf16f(w[i + 1]) * x[i + 1];
                s2 += bf16f(w[i + 2]) * x[i + 2];
                s3 += bf16f(w[i + 3]) * x[i + 3];
            }
            for (; i < cols; i++) s0 += bf16f(w[i]) * x[i];
            out[r] = (s0 + s1) + (s2 + s3);
        }
    }
}

static void rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * inv * w[i]; /* safe in place */
}

static void softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

static float silu(float x) { return x / (1.0f + expf(-x)); }

/* Rotate pairs (i, i+half) -- same convention as the reference. */
static void rope(float *v, int heads, int head_dim, int pos, float theta) {
    int half = head_dim / 2;
    for (int h = 0; h < heads; h++) {
        float *p = v + (size_t)h * head_dim;
        for (int i = 0; i < half; i++) {
            double freq = pow((double)theta, -2.0 * i / head_dim);
            double ang = pos * freq;
            float cs = (float)cos(ang), sn = (float)sin(ang);
            float a = p[i], b = p[i + half];
            p[i] = a * cs - b * sn;
            p[i + half] = b * cs + a * sn;
        }
    }
}

/* ------------------------------------------------------------------ forward pass */

static void state_init(State *s, const Config *c, int ctx) {
    int q_dim = c->n_heads * c->head_dim, kv_dim = c->n_kv_heads * c->head_dim;
    s->ctx = ctx;
    s->x = (float *)calloc((size_t)c->dim, sizeof(float));
    s->xb = (float *)calloc((size_t)c->dim, sizeof(float));
    s->q = (float *)calloc((size_t)q_dim, sizeof(float));
    s->k = (float *)calloc((size_t)kv_dim, sizeof(float));
    s->v = (float *)calloc((size_t)kv_dim, sizeof(float));
    s->att = (float *)calloc((size_t)ctx, sizeof(float));
    s->attn_out = (float *)calloc((size_t)q_dim, sizeof(float));
    s->router = (float *)calloc((size_t)c->n_experts, sizeof(float));
    s->h1 = (float *)calloc((size_t)c->expert_hidden, sizeof(float));
    s->h3 = (float *)calloc((size_t)c->expert_hidden, sizeof(float));
    s->tmp = (float *)calloc((size_t)c->dim, sizeof(float));
    s->moe = (float *)calloc((size_t)c->dim, sizeof(float));
    s->logits = (float *)calloc((size_t)c->vocab, sizeof(float));
    s->kcache = (float *)calloc((size_t)c->n_layers * ctx * kv_dim, sizeof(float));
    s->vcache = (float *)calloc((size_t)c->n_layers * ctx * kv_dim, sizeof(float));
    s->used = (unsigned char *)calloc((size_t)c->n_experts, 1);
}

static int state_ok(const State *s) {
    return s->x && s->xb && s->q && s->k && s->v && s->att && s->attn_out && s->router && s->h1 &&
           s->h3 && s->tmp && s->moe && s->logits && s->kcache && s->vcache && s->used;
}

/* Processes one token at position pos; leaves logits in s->logits.
 * routing_out receives the chosen experts, [n_layers][top_k]. */
static int forward(Model *m, State *s, int token, int pos, int32_t *routing_out) {
    const Config *c = &m->c;
    const int hd = c->head_dim, q_dim = c->n_heads * hd, kv_dim = c->n_kv_heads * hd;
    const int group = c->n_heads / c->n_kv_heads;
    const int H = c->expert_hidden, D = c->dim;

    /* embedding row */
    if (c->dtype == DT_F32) {
        memcpy(s->x, (const float *)MAT(m, m->tok_emb) + (size_t)token * D, (size_t)D * sizeof(float));
    } else {
        const uint16_t *row = (const uint16_t *)MAT(m, m->tok_emb) + (size_t)token * D;
        for (int i = 0; i < D; i++) s->x[i] = bf16f(row[i]);
    }

    for (int l = 0; l < c->n_layers; l++) {
        const LayerOff *ly = &m->layers[l];

        /* ---- attention ---- */
        rmsnorm(s->xb, s->x, VEC(m, ly->attn_norm), D, c->rms_eps);
        matvec(s->q, MAT(m, ly->wq), c->dtype, s->xb, q_dim, D);
        matvec(s->k, MAT(m, ly->wk), c->dtype, s->xb, kv_dim, D);
        matvec(s->v, MAT(m, ly->wv), c->dtype, s->xb, kv_dim, D);
        if (c->qk_norm) { /* over the WHOLE projection, before splitting into heads (OLMoE) */
            rmsnorm(s->q, s->q, VEC(m, ly->q_norm), q_dim, c->rms_eps);
            rmsnorm(s->k, s->k, VEC(m, ly->k_norm), kv_dim, c->rms_eps);
        }
        rope(s->q, c->n_heads, hd, pos, c->rope_theta);
        rope(s->k, c->n_kv_heads, hd, pos, c->rope_theta);

        float *kc = s->kcache + ((size_t)l * s->ctx) * kv_dim;
        float *vc = s->vcache + ((size_t)l * s->ctx) * kv_dim;
        memcpy(kc + (size_t)pos * kv_dim, s->k, (size_t)kv_dim * sizeof(float));
        memcpy(vc + (size_t)pos * kv_dim, s->v, (size_t)kv_dim * sizeof(float));

        const float scale = 1.0f / sqrtf((float)hd);
        for (int h = 0; h < c->n_heads; h++) {
            const float *qh = s->q + (size_t)h * hd;
            const int kvh = h / group;
            for (int t = 0; t <= pos; t++) {
                const float *kt = kc + (size_t)t * kv_dim + (size_t)kvh * hd;
                float dot = 0.0f;
                for (int i = 0; i < hd; i++) dot += qh[i] * kt[i];
                s->att[t] = dot * scale;
            }
            softmax(s->att, pos + 1);
            float *out = s->attn_out + (size_t)h * hd;
            memset(out, 0, (size_t)hd * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                const float *vt = vc + (size_t)t * kv_dim + (size_t)kvh * hd;
                const float a = s->att[t];
                for (int i = 0; i < hd; i++) out[i] += a * vt[i];
            }
        }
        matvec(s->xb, MAT(m, ly->wo), c->dtype, s->attn_out, D, q_dim); /* xb reused as scratch */
        for (int i = 0; i < D; i++) s->x[i] += s->xb[i];

        /* ---- MoE feed-forward ---- */
        rmsnorm(s->xb, s->x, VEC(m, ly->ffn_norm), D, c->rms_eps);
        matvec(s->router, MAT(m, ly->router), c->dtype, s->xb, c->n_experts, D);
        softmax(s->router, c->n_experts);

        int idx[MAX_TOPK];
        float wt[MAX_TOPK], wsum = 0.0f;
        memset(s->used, 0, (size_t)c->n_experts);
        for (int k = 0; k < c->top_k; k++) {
            int best = -1;
            for (int e = 0; e < c->n_experts; e++)
                if (!s->used[e] && (best < 0 || s->router[e] > s->router[best])) best = e;
            s->used[best] = 1;
            idx[k] = best;
            wt[k] = s->router[best];
            wsum += wt[k];
            routing_out[(size_t)l * c->top_k + k] = best;
        }

        memset(s->moe, 0, (size_t)D * sizeof(float));
        for (int k = 0; k < c->top_k; k++) {
            const unsigned char *eb = get_expert(m, l, idx[k]); /* <-- the streaming seam */
            if (!eb) return 0;
            const size_t mat_bytes = (size_t)H * D * m->esz;
            matvec(s->h1, eb, c->dtype, s->xb, H, D);                   /* w1 (gate) */
            matvec(s->h3, eb + mat_bytes, c->dtype, s->xb, H, D);       /* w3 (up)   */
            for (int i = 0; i < H; i++) s->h1[i] = silu(s->h1[i]) * s->h3[i];
            matvec(s->tmp, eb + 2 * mat_bytes, c->dtype, s->h1, D, H);  /* w2 (down) */
            const float wk = c->norm_topk ? wt[k] / wsum : wt[k];
            for (int i = 0; i < D; i++) s->moe[i] += wk * s->tmp[i];
        }
        for (int i = 0; i < D; i++) s->x[i] += s->moe[i];
    }

    rmsnorm(s->xb, s->x, VEC(m, m->final_norm), D, c->rms_eps);
    matvec(s->logits, MAT(m, m->lm_head), c->dtype, s->xb, c->vocab, D);
    return 1;
}

/* ------------------------------------------------------------------ main */

static int argmax(const float *v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

static void usage(const char *p) {
    fprintf(stderr,
            "usage:\n"
            "  %s <model.rbt2> --dump <logits.bin> <routing.bin> [--ctx N] [--expert-cache-mib N] [--kernel auto|scalar|avx2] <token> ...\n"
            "  %s <model.rbt2> --gen <n> [--eos id] [--ctx N] [--expert-cache-mib N] [--kernel auto|scalar|avx2] <token> ...\n"
            "  %s <model.rbt2> --info\n", p, p, p);
}

int main(int argc, char **argv) {
    if (argc < 3) { usage(argv[0]); return 2; }

    const char *dump_l = NULL, *dump_r = NULL, *kernel = NULL;
    int gen = 0, eos = -1, ctx = 0, nt = 0, info = 0;
    size_t expert_cache_budget = 0;
    int *tokens = (int *)malloc(sizeof(int) * (size_t)argc);
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--dump") && i + 2 < argc) { dump_l = argv[++i]; dump_r = argv[++i]; }
        else if (!strcmp(argv[i], "--info")) info = 1;
        else if (!strcmp(argv[i], "--gen") && i + 1 < argc) gen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--eos") && i + 1 < argc) eos = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--kernel") && i + 1 < argc) kernel = argv[++i];
        else if (!strcmp(argv[i], "--expert-cache-mib") && i + 1 < argc) {
            const char *arg = argv[++i];
            char *end = NULL;
            errno = 0;
            unsigned long long mib = strtoull(arg, &end, 10);
            if (arg[0] < '0' || arg[0] > '9' || *end || errno || !mib || mib > SIZE_MAX / (1024u * 1024u)) {
                fprintf(stderr, "--expert-cache-mib must be a positive integer that fits in memory\n"); return 2;
            }
            expert_cache_budget = (size_t)mib * (1024u * 1024u);
        }
        else if (argv[i][0] == '-' && argv[i][1] == '-') { fprintf(stderr, "bad option %s\n", argv[i]); usage(argv[0]); return 2; }
        else tokens[nt++] = atoi(argv[i]);
    }
    if (!info && (nt < 1 || (!dump_l && gen <= 0))) { usage(argv[0]); return 2; }
    if (kernel) {
        if (strcmp(kernel, "scalar") && strcmp(kernel, "auto") && strcmp(kernel, "avx2")) {
            fprintf(stderr, "--kernel must be scalar, auto or avx2\n"); return 2;
        }
        if (!strcmp(kernel, "avx2") && !cpu_has_avx2()) {
            fprintf(stderr, "AVX2 kernel unavailable on this CPU or build\n"); return 1;
        }
        use_avx2 = strcmp(kernel, "scalar") != 0 && cpu_has_avx2();
        fprintf(stderr, "matvec kernel: %s\n", use_avx2 ? "avx2" : "scalar");
    }

    Model m;
    memset(&m, 0, sizeof(m));
    if (!load_model(&m, argv[1])) return 1;
    const Config *c = &m.c;

    if (info) { /* used by tests to check the C layout against ref/rbt2.py at full model scale */
        printf("vocab=%d dim=%d n_layers=%d n_heads=%d n_kv_heads=%d head_dim=%d n_experts=%d top_k=%d\n",
               c->vocab, c->dim, c->n_layers, c->n_heads, c->n_kv_heads, c->head_dim, c->n_experts, c->top_k);
        printf("expert_hidden=%d max_seq=%d dtype=%d qk_norm=%d norm_topk=%d\n",
               c->expert_hidden, c->max_seq, c->dtype, c->qk_norm, c->norm_topk);
        printf("tok_emb=%llu layer0_experts=%llu last_layer_experts=%llu expert_stride=%llu lm_head=%llu "
               "layout_end=%llu file_size=%llu\n",
               (unsigned long long)m.tok_emb, (unsigned long long)m.layers[0].experts,
               (unsigned long long)m.layers[c->n_layers - 1].experts, (unsigned long long)m.expert_stride,
               (unsigned long long)m.lm_head, (unsigned long long)layout_end(&m),
               (unsigned long long)m.map.size);
        return 0;
    }

    if (expert_cache_budget && !expert_cache_init(&m, argv[1], expert_cache_budget)) return 1;

    if (ctx <= 0) ctx = c->max_seq < 512 ? c->max_seq : 512;
    if (ctx > c->max_seq) ctx = c->max_seq;
    if (nt > ctx) { fprintf(stderr, "prompt has %d tokens but context is %d (use --ctx)\n", nt, ctx); return 1; }
    for (int i = 0; i < nt; i++)
        if (tokens[i] < 0 || tokens[i] >= c->vocab) { fprintf(stderr, "token %d out of range\n", tokens[i]); return 1; }

    State s;
    memset(&s, 0, sizeof(s));
    state_init(&s, c, ctx);
    if (!state_ok(&s)) { fprintf(stderr, "out of memory (try a smaller --ctx)\n"); return 1; }
    int32_t *routing = (int32_t *)calloc((size_t)c->n_layers * c->top_k, sizeof(int32_t));

    FILE *fl = NULL, *fr = NULL;
    if (dump_l) {
        fl = fopen(dump_l, "wb");
        fr = fopen(dump_r, "wb");
        if (!fl || !fr) { fprintf(stderr, "cannot open output files\n"); return 1; }
    }

    double t0 = now_sec();
    for (int i = 0; i < nt; i++) {
        if (!forward(&m, &s, tokens[i], i, routing)) {
            if (fl) { fclose(fl); fclose(fr); }
            expert_cache_close(&m.cache);
            return 1;
        }
        if (fl) {
            fwrite(s.logits, sizeof(float), (size_t)c->vocab, fl);
            fwrite(routing, sizeof(int32_t), (size_t)c->n_layers * c->top_k, fr);
        }
    }
    double t1 = now_sec();
    if (fl) { fclose(fl); fclose(fr); }

    int produced = 0;
    if (gen > 0) {
        if (gen > ctx - nt) gen = ctx - nt;
        int pos = nt;
        for (int g = 0; g < gen; g++) {
            int next = argmax(s.logits, c->vocab);
            printf("%d\n", next);
            fflush(stdout);
            produced++;
            if (next == eos) break;
            if (g + 1 == gen) break;
            if (!forward(&m, &s, next, pos++, routing)) {
                expert_cache_close(&m.cache);
                return 1;
            }
        }
    }
    double t2 = now_sec();
    fprintf(stderr, "prompt: %d tokens in %.2fs (%.2f tok/s)", nt, t1 - t0, nt / (t1 - t0 + 1e-9));
    if (gen > 0) fprintf(stderr, " | generated %d in %.2fs (%.2f tok/s)", produced, t2 - t1, produced / (t2 - t1 + 1e-9));
    fprintf(stderr, "\n");
    if (m.cache.file) {
        fprintf(stderr, "expert cache: hits=%llu misses=%llu evictions=%llu bytes_read=%llu "
                "resident_bytes=%llu capacity_bytes=%llu\n",
                (unsigned long long)m.cache.hits, (unsigned long long)m.cache.misses,
                (unsigned long long)m.cache.evictions, (unsigned long long)m.cache.bytes_read,
                (unsigned long long)(m.cache.resident * m.expert_stride),
                (unsigned long long)(m.cache.capacity * m.expert_stride));
        expert_cache_close(&m.cache);
    }
    return 0;
}
