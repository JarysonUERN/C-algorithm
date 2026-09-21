#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define BASE          100000u     
#define BASE_DIGITS   5
#define SCHOOL_LIMIT  32           
#define LEAF_SIZE     3            
#define MAX_N         2500000u     

#define MOD1 998244353u            
#define MOD2 469762049u            
#define ROOT 3u
#define AINLINE static inline __attribute__((always_inline))

typedef struct {
    uint32_t *d;   
    size_t    n;   
} BigInt;

static void *xmalloc(size_t bytes)
{
    void *p = malloc(bytes ? bytes : 1);
    if (!p) { fprintf(stderr, "Erro: memoria insuficiente\n"); exit(1); }
    return p;
}

static void *xcalloc(size_t cnt, size_t sz)
{
    void *p = calloc(cnt ? cnt : 1, sz);
    if (!p) { fprintf(stderr, "Erro: memoria insuficiente\n"); exit(1); }
    return p;
}

static double agora(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static uint32_t mod_pow(uint64_t b, uint64_t e, uint32_t m)
{
    uint64_t r = 1;
    b %= m;
    while (e) {
        if (e & 1) r = r * b % m;
        b = b * b % m;
        e >>= 1;
    }
    return (uint32_t)r;
}

typedef struct {
    uint32_t *fw;     
    uint32_t *iv;     
    size_t    size;  
} Roots;

static Roots R1, R2;
static uint32_t INV_P1_MOD_P2;

/* Garante tabelas para transformadas de tamanho n (potência de 2). */
static void ensure_roots(Roots *R, size_t n, uint32_t mod)
{
    if (R->size == 0) {
        R->fw = xmalloc(2 * sizeof(uint32_t));
        R->iv = xmalloc(2 * sizeof(uint32_t));
        R->fw[0] = R->iv[0] = 0;
        R->fw[1] = R->iv[1] = 1;
        R->size = 2;
    }
    while (R->size < n) {
        size_t s = R->size;                 /* half = s, len = 2s */
        R->fw = realloc(R->fw, 2 * s * sizeof(uint32_t));
        R->iv = realloc(R->iv, 2 * s * sizeof(uint32_t));
        if (!R->fw || !R->iv) { fprintf(stderr, "Erro: memoria\n"); exit(1); }
        uint32_t w  = mod_pow(ROOT, (mod - 1) / (2 * s), mod);
        uint32_t wi = mod_pow(w, mod - 2, mod);
        uint64_t a = 1, b = 1;
        for (size_t j = 0; j < s; j++) {
            R->fw[s + j] = (uint32_t)a;
            R->iv[s + j] = (uint32_t)b;
            a = a * w % mod;
            b = b * wi % mod;
        }
        R->size = 2 * s;
    }
}

AINLINE void ntt_fwd_impl(uint32_t *a, size_t n, const uint32_t *rt, const uint32_t mod)
{
    for (size_t half = n >> 1; half >= 1; half >>= 1) {
        for (size_t i = 0; i < n; i += 2 * half) {
            uint32_t *x = a + i, *y = a + i + half;
            const uint32_t *w = rt + half;
            for (size_t j = 0; j < half; j++) {
                uint32_t u = x[j], v = y[j];
                uint32_t s = u + v;        if (s >= mod) s -= mod;
                uint32_t d = u + mod - v;  if (d >= mod) d -= mod;
                x[j] = s;
                y[j] = (uint32_t)((uint64_t)d * w[j] % mod);
            }
        }
    }
}

AINLINE void ntt_inv_impl(uint32_t *a, size_t n, const uint32_t *rt, const uint32_t mod)
{
    for (size_t half = 1; half < n; half <<= 1) {
        for (size_t i = 0; i < n; i += 2 * half) {
            uint32_t *x = a + i, *y = a + i + half;
            const uint32_t *w = rt + half;
            for (size_t j = 0; j < half; j++) {
                uint32_t u = x[j];
                uint32_t v = (uint32_t)((uint64_t)y[j] * w[j] % mod);
                uint32_t s = u + v;        if (s >= mod) s -= mod;
                uint32_t d = u + mod - v;  if (d >= mod) d -= mod;
                x[j] = s;
                y[j] = d;
            }
        }
    }
}

static void ntt_fwd1(uint32_t *a, size_t n) { ntt_fwd_impl(a, n, R1.fw, MOD1); }
    static void ntt_inv1(uint32_t *a, size_t n) { ntt_inv_impl(a, n, R1.iv, MOD1); }
    static void ntt_fwd2(uint32_t *a, size_t n) { ntt_fwd_impl(a, n, R2.fw, MOD2); }
static void ntt_inv2(uint32_t *a, size_t n) { ntt_inv_impl(a, n, R2.iv, MOD2); }



static void pointwise1(uint32_t *a, const uint32_t *b, size_t n, uint32_t ninv)
{
    for (size_t i = 0; i < n; i++) {
        uint64_t t = (uint64_t)a[i] * b[i] % MOD1;
        a[i] = (uint32_t)(t * ninv % MOD1);
    }
}

static void pointwise2(uint32_t *a, const uint32_t *b, size_t n, uint32_t ninv)
{
    for (size_t i = 0; i < n; i++) {
            uint64_t t = (uint64_t)a[i] * b[i] % MOD2;
            a[i] = (uint32_t)(t * ninv % MOD2);
    }
}

static void big_free(BigInt *x)
{
    free(x->d);
    x->d = NULL;
    x->n = 0;
}

static void big_trim(BigInt *x)
{
    while (x->n > 1 && x->d[x->n - 1] == 0) x->n--;
}
static BigInt big_from_u64(uint64_t v)
{
    BigInt r;
    r.d = xmalloc(4 * sizeof(uint32_t));   
    r.n = 0;
    do {
        r.d[r.n++] = (uint32_t)(v % BASE);
        v /= BASE;
    } while (v);
    return r;
}
static BigInt mul_school(const BigInt *a, const BigInt *b)
{
    size_t rn = a->n + b->n;
    uint64_t *t = xcalloc(rn, sizeof(uint64_t));
    for (size_t i = 0; i < a->n; i++) {
        uint64_t ai = a->d[i];
        if (!ai) continue;
        for (size_t j = 0; j < b->n; j++)
            t[i + j] += ai * b->d[j];
    }
    BigInt r;
    r.d = xmalloc(rn * sizeof(uint32_t));
    r.n = rn;
    uint64_t c = 0;
    for (size_t k = 0; k < rn; k++) {
        uint64_t v = t[k] + c;
        r.d[k] = (uint32_t)(v % BASE);
        c = v / BASE;
    }
    free(t);
    big_trim(&r);
    return r;
}

static uint32_t *g_scratch = NULL;
static size_t    g_scratch_cap = 0;

static uint32_t *scratch_get(size_t elems)
{
    if (elems > g_scratch_cap) {
        free(g_scratch);
        g_scratch = xmalloc(elems * sizeof(uint32_t));
        g_scratch_cap = elems;
    }
    return g_scratch;
}

static BigInt mul_ntt(const BigInt *a, const BigInt *b)
{
    size_t rn = a->n + b->n;
    size_t n = 1;
    while (n < rn) n <<= 1;

    ensure_roots(&R1, n, MOD1);
    ensure_roots(&R2, n, MOD2);

    uint32_t *buf = scratch_get(4 * n);
    uint32_t *A1 = buf, *B1 = buf + n, *A2 = buf + 2 * n, *B2 = buf + 3 * n;

    memcpy(A1, a->d, a->n * sizeof(uint32_t));
    memset(A1 + a->n, 0, (n - a->n) * sizeof(uint32_t));
    memcpy(B1, b->d, b->n * sizeof(uint32_t));
    memset(B1 + b->n, 0, (n - b->n) * sizeof(uint32_t));
    memcpy(A2, A1, n * sizeof(uint32_t));
    memcpy(B2, B1, n * sizeof(uint32_t));

    uint32_t ninv1 = mod_pow(n, MOD1 - 2, MOD1);
    uint32_t ninv2 = mod_pow(n, MOD2 - 2, MOD2);

    ntt_fwd1(A1, n); ntt_fwd1(B1, n); pointwise1(A1, B1, n, ninv1); ntt_inv1(A1, n);
    ntt_fwd2(A2, n); ntt_fwd2(B2, n); pointwise2(A2, B2, n, ninv2); ntt_inv2(A2, n);

    BigInt r;
    r.d = xmalloc(rn * sizeof(uint32_t));
    r.n = rn;

    /* CRT: x = x1 + MOD1 * ((x2 - x1) * INV_P1_MOD_P2 mod MOD2) */
    uint64_t carry = 0;
    for (size_t k = 0; k < rn; k++) {
        uint32_t x1 = A1[k], x2 = A2[k];
        uint32_t d = x2 + MOD2 - (x1 % MOD2);
        if (d >= MOD2) d -= MOD2;
        uint32_t t = (uint32_t)((uint64_t)d * INV_P1_MOD_P2 % MOD2);
        uint64_t x = (uint64_t)x1 + (uint64_t)MOD1 * t + carry;
        r.d[k] = (uint32_t)(x % BASE);
        carry = x / BASE;
    }
    big_trim(&r);
    return r;
}

static BigInt big_mul(const BigInt *a, const BigInt *b)
{
    if (a->n < SCHOOL_LIMIT || b->n < SCHOOL_LIMIT)
        return mul_school(a, b);
    return mul_ntt(a, b);
}

static BigInt product_range(uint32_t lo, uint32_t hi)
{
    uint32_t cnt = hi - lo + 1;
    // int use_ntt = cnt >= 2 * SCHOOL_LIMIT; 
    if (cnt <= LEAF_SIZE) {
        uint64_t p = 1;
        for (uint32_t i = lo; i <= hi; i++) p *= i;
        return big_from_u64(p);
    }

    uint32_t mid = lo + (hi - lo) / 2;
    BigInt left  = product_range(lo, mid);
    BigInt right = product_range(mid + 1, hi);
    BigInt res   = big_mul(&left, &right);
        //big_free(&left);
    big_free(&left);
    big_free(&right);
    return res;
}

static BigInt factorial(uint32_t n)
{
    if (n < 2) return big_from_u64(1);
    return product_range(2, n);
}

static char *to_decimal_buffer(const BigInt *x, size_t *len)
{
    char *buf = xmalloc(x->n * BASE_DIGITS + 2);
    char *p = buf;
        uint32_t top = x->d[x->n - 1];
    
    char tmp[BASE_DIGITS];
    int t = 0;
        //for (;;) { tmp[t++] = (char)('0' + top % 10); top /= 10; if (!top) break; }
    do { tmp[t++] = (char)('0' + top % 10); top /= 10; } while (top);
    while (t) *p++ = tmp[--t];

    for (size_t i = x->n - 1; i-- > 0; ) {
        uint32_t v = x->d[i];
        p[4] = (char)('0' + v % 10); v /= 10;
        p[3] = (char)('0' + v % 10); v /= 10;
        p[2] = (char)('0' + v % 10); v /= 10;
        p[1] = (char)('0' + v % 10); v /= 10;
        p[0] = (char)('0' + v);
        p += 5;
    }
    *p = '\0';
    *len = (size_t)(p - buf);
    return buf;
}

static int write_buffer(const char *path, const char *buf, size_t len)
{
    FILE *f = fopen(path, "wb");
    if (!f) { perror("fopen"); return 0; }
    setvbuf(f, NULL, _IOFBF, 1 << 20);
    size_t w = fwrite(buf, 1, len, f);
    fputc('\n', f);
    fclose(f);
    return w == len;
}

static void print_summary(const char *buf, size_t len)
{
    size_t zeros = 0;
    while (zeros < len && buf[len - 1 - zeros] == '0') zeros++;

    size_t show = len < 20 ? len : 20;
    printf("Digitos            : %lu\n", (unsigned long)len);
    printf("Zeros no final     : %lu\n", (unsigned long)zeros);
    printf("Primeiros %lu dig.  : %.*s\n", (unsigned long)show, (int)show, buf);
    if (len > 40) {
        printf("Ultimos 20 dig    : %s\n", buf + len - 20);
    }
}

int main(int argc, char **argv)
{
    uint32_t n = 1000000;
    const char *out = NULL;
    char defname[64];

    if (argc > 1) {
        long v = atol(argv[1]);
        if (v < 0 || (unsigned long)v > MAX_N) {
            fprintf(stderr, "N deve estar entre 0 e %u\n", MAX_N);
            return 1;
        }
        n = (uint32_t)v;
    }
    if (argc > 2) out = argv[2];
    else { snprintf(defname, sizeof defname, "fatorial_%u.txt", n); out = defname; }
            printf("Calculando fatorial de %u ...\n", n); // garantir que ta funcionando
    INV_P1_MOD_P2 = mod_pow(MOD1 % MOD2, MOD2 - 2, MOD2);

    double t0 = agora();
    BigInt f = factorial(n);
    double t1 = agora();

    size_t len;
    char *buf = to_decimal_buffer(&f, &len);
    double t2 = agora();

    int ok = write_buffer(out, buf, len);
    double t3 = agora();

    printf("Fatorial de %u\n", n);
    print_summary(buf, len);
    printf("Tempo calculo      : %.3f s\n", t1 - t0);
    printf("Tempo conversao    : %.3f s\n", t2 - t1);
    printf("Tempo escrita      : %.3f s\n", t3 - t2);
    if (ok) printf("Guardado em        : %s\n", out);

    free(buf);
    big_free(&f);
    free(g_scratch);
    free(R1.fw); free(R1.iv); free(R2.fw); free(R2.iv);
    return ok ? 0 : 1;
}