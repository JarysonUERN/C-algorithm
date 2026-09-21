#include <stdio.h>
#include <stdlib.h>
// #include <stdint.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define MR 4        
#define KC 256      
#define NC 512      
#define OUT_BUF_SIZE (1u << 20)
#define PRINT_LIMIT  400          

typedef struct {
    double *d;
    size_t  r, c;   
} Matrix;

#define AT(M, i, j) ((M)->d[(size_t)(i) * (M)->c + (size_t)(j)])

static double agora(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void *alloc_aligned(size_t bytes)
{
    if (bytes == 0) bytes = 64;
    bytes = (bytes + 63) & ~(size_t)63;         
    return aligned_alloc(64, bytes);
}

static int mat_alloc(Matrix *M, size_t r, size_t c)
{
    M->d = NULL; M->r = r; M->c = c;
    if (r != 0 && c > SIZE_MAX / sizeof(double) / r) return 0;  
    M->d = alloc_aligned(r * c * sizeof(double));
    return M->d != NULL;
}

static void mat_free(Matrix *M)
{
    free(M->d);
    M->d = NULL;
}

static void kernel4(double *restrict c0, double *restrict c1,
                    double *restrict c2, double *restrict c3,
                    const double *a0, const double *a1,
                    const double *a2, const double *a3,
                    const double *restrict Bp, size_t kc, size_t nc)
{
    size_t p = 0;
    for (; p + 4 <= kc; p += 4) {
        const double *b0 = Bp + p * nc, *b1 = b0 + nc, *b2 = b1 + nc, *b3 = b2 + nc;
        const double a00 = a0[p], a01 = a0[p+1], a02 = a0[p+2], a03 = a0[p+3];
        const double a10 = a1[p], a11 = a1[p+1], a12 = a1[p+2], a13 = a1[p+3];
        const double a20 = a2[p], a21 = a2[p+1], a22 = a2[p+2], a23 = a2[p+3];
        const double a30 = a3[p], a31 = a3[p+1], a32 = a3[p+2], a33 = a3[p+3];
        for (size_t j = 0; j < nc; j++) {
            const double x0 = b0[j], x1 = b1[j], x2 = b2[j], x3 = b3[j];
            c0[j] += a00 * x0 + a01 * x1 + a02 * x2 + a03 * x3;
            c1[j] += a10 * x0 + a11 * x1 + a12 * x2 + a13 * x3;
            c2[j] += a20 * x0 + a21 * x1 + a22 * x2 + a23 * x3;
            c3[j] += a30 * x0 + a31 * x1 + a32 * x2 + a33 * x3;
        }
    }
    for (; p < kc; p++) {
        const double *b = Bp + p * nc;
        const double s0 = a0[p], s1 = a1[p], s2 = a2[p], s3 = a3[p];
        for (size_t j = 0; j < nc; j++) {
            const double x = b[j];
            c0[j] += s0 * x; c1[j] += s1 * x; c2[j] += s2 * x; c3[j] += s3 * x;
        }
    }
}

static void kernel1(double *restrict c, const double *a,
                    const double *restrict Bp, size_t kc, size_t nc)
{
    size_t p = 0;
    for (; p + 4 <= kc; p += 4) {
        const double *b0 = Bp + p * nc, *b1 = b0 + nc, *b2 = b1 + nc, *b3 = b2 + nc;
        const double s0 = a[p], s1 = a[p+1], s2 = a[p+2], s3 = a[p+3];
        for (size_t j = 0; j < nc; j++)
            c[j] += s0 * b0[j] + s1 * b1[j] + s2 * b2[j] + s3 * b3[j];
    }
    for (; p < kc; p++) {
        const double *b = Bp + p * nc;
        const double s = a[p];
        for (size_t j = 0; j < nc; j++) c[j] += s * b[j];
    }
}

static int mat_mul(const Matrix *A, const Matrix *B, Matrix *C)
{
    const size_t m = A->r, k = A->c, n = B->c;
    if (!mat_alloc(C, m, n)) return 0;
    memset(C->d, 0, m * n * sizeof(double));
    if (m == 0 || n == 0 || k == 0) return 1;

    double *Bp = alloc_aligned((size_t)KC * NC * sizeof(double));   
    if (!Bp) { mat_free(C); return 0; }

    for (size_t jc = 0; jc < n; jc += NC) {
        const size_t nc = (n - jc < NC) ? n - jc : NC;
        for (size_t pc = 0; pc < k; pc += KC) {
            const size_t kc = (k - pc < KC) ? k - pc : KC;

            for (size_t p = 0; p < kc; p++)
                memcpy(Bp + p * nc, B->d + (pc + p) * n + jc, nc * sizeof(double));

            #pragma omp parallel for schedule(static)
            for (size_t ic = 0; ic < m; ic += MR) {
                double *c = C->d + ic * n + jc;
                const double *a = A->d + ic * k + pc;
                if (ic + MR <= m) {
                    kernel4(c, c + n, c + 2 * n, c + 3 * n,
                            a, a + k, a + 2 * k, a + 3 * k, Bp, kc, nc);
                } else {
                    for (size_t r = ic; r < m; r++)
                        kernel1(C->d + r * n + jc, A->d + r * k + pc, Bp, kc, nc);
                }
            }
        }
    }
    free(Bp);
    return 1;
}

static int mat_hadamard(const Matrix *A, const Matrix *B, Matrix *C)
{
    if (!mat_alloc(C, A->r, A->c)) return 0;
    const size_t total = A->r * A->c;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < total; i++) C->d[i] = A->d[i] * B->d[i];
    return 1;
}

static int mat_kronecker(const Matrix *A, const Matrix *B, Matrix *C)
{
    const size_t R = A->r * B->r, Cc = A->c * B->c;
    if (A->r && R / A->r != B->r) return 0;
    if (A->c && Cc / A->c != B->c) return 0;
    if (!mat_alloc(C, R, Cc)) return 0;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < A->r; i++)
        for (size_t p = 0; p < B->r; p++) {
            double *dst = C->d + (i * B->r + p) * Cc;
            const double *brow = B->d + p * B->c;
            for (size_t j = 0; j < A->c; j++) {
                const double a = A->d[i * A->c + j];
                for (size_t q = 0; q < B->c; q++)
                    dst[j * B->c + q] = a * brow[q];
            }
        }
    return 1;
}
static void mat_mul_naive(const Matrix *A, const Matrix *B, Matrix *C)
{
    for (size_t i = 0; i < A->r; i++)
        for (size_t j = 0; j < B->c; j++) {
            double s = 0;
            for (size_t p = 0; p < A->c; p++) s += AT(A, i, p) * AT(B, p, j);
            AT(C, i, j) = s;
        }
}

typedef struct {
    FILE  *f;
    char  *buf;
    size_t pos;
} OutBuf;

static void ob_flush(OutBuf *o)
{
    if (o->pos) { fwrite(o->buf, 1, o->pos, o->f); o->pos = 0; }
}

static void ob_put_double(OutBuf *o, double v, char sep)
{
    if (o->pos + 40 > OUT_BUF_SIZE) ob_flush(o);
    int n = snprintf(o->buf + o->pos, 32, "%.10g", v);
    o->pos += (size_t)n;
    o->buf[o->pos++] = sep;
}

static void print_matrix(FILE *f, const Matrix *M)
{
    OutBuf o = { f, malloc(OUT_BUF_SIZE), 0 };
    if (!o.buf) { fprintf(stderr, "Sem memoria para o buffer\n"); return; }
    for (size_t i = 0; i < M->r; i++)
        for (size_t j = 0; j < M->c; j++)
            ob_put_double(&o, AT(M, i, j), j + 1 == M->c ? '\n' : ' ');
    ob_flush(&o);
    free(o.buf);
}

static int ask_long(const char *prompt, long lo, long hi, long *out)
{
    for (;;) {
        printf("%s", prompt);
        fflush(stdout);
        if (scanf("%ld", out) != 1) {
            int ch;
            if (feof(stdin)) return 0;
            while ((ch = getchar()) != '\n' && ch != EOF) {}
            printf("Valor invalido.\n");
            continue;
        }
        if (*out >= lo && *out <= hi) return 1;
        printf("Deve estar entre %ld e %ld.\n", lo, hi);
    }
}

static int ask_double(const char *prompt, double *out)
{
    printf("%s", prompt);
    fflush(stdout);
    return scanf("%lf", out) == 1;
}

static int load_from_file(Matrix *M, const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror("fopen"); return 0; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) { fclose(f); return 0; }
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return 0; }
    size_t got = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[got] = '\0';

    const size_t total = M->r * M->c;
    char *p = buf;
    size_t i = 0;
    for (; i < total; i++) {
        char *end;
        double v = strtod(p, &end);
        if (end == p) break;
        M->d[i] = v;
        p = end;
    }
    free(buf);
    if (i < total) {
        fprintf(stderr, "O ficheiro so tem %zu de %zu valores.\n", i, total);
        return 0;
    }
    return 1;
}

static uint64_t rng_state;
static double rand_range(double lo, double hi)
{
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    uint64_t r = rng_state * 0x2545F4914F6CDD1DULL;
    double u = (double)(r >> 11) * (1.0 / 9007199254740992.0);
    return lo + u * (hi - lo);
}

static int read_matrix(Matrix *M, const char *nome, long fix_r, long fix_c)
{
    long r, c, src;
    char prompt[96];
    const long MAXDIM = 100000;

    printf("\n--- Matriz %s ---\n", nome);
    if (fix_r > 0) { r = fix_r; printf("Linhas: %ld (imposto pela operacao)\n", r); }
    else {
        snprintf(prompt, sizeof prompt, "Linhas de %s: ", nome);
        if (!ask_long(prompt, 1, MAXDIM, &r)) return 0;
    }
    if (fix_c > 0) { c = fix_c; printf("Colunas: %ld (imposto pela operacao)\n", c); }
    else {
        snprintf(prompt, sizeof prompt, "Colunas de %s: ", nome);
        if (!ask_long(prompt, 1, MAXDIM, &c)) return 0;
    }
    if (!mat_alloc(M, (size_t)r, (size_t)c)) {
        fprintf(stderr, "Sem memoria para uma matriz %ldx%ld.\n", r, c);
        return 0;
    }

    if (!ask_long("Origem dos dados: 1) digitar  2) aleatorios  3) ficheiro: ", 1, 3, &src))
        return 0;

    const size_t total = M->r * M->c;
    if (src == 1) {
        printf("Introduza %zu valores (linha a linha):\n", total);
        for (size_t i = 0; i < total; i++)
            if (scanf("%lf", &M->d[i]) != 1) {
                fprintf(stderr, "Entrada invalida.\n");
                return 0;
            }
    } else if (src == 2) {
        double lo, hi;
        if (!ask_double("Minimo: ", &lo) || !ask_double("Maximo: ", &hi)) return 0;
        for (size_t i = 0; i < total; i++) M->d[i] = rand_range(lo, hi);
    } else {
        char path[512];
        printf("Caminho do ficheiro (%zu valores separados por espacos/linhas): ", total);
        fflush(stdout);
        if (scanf("%511s", path) != 1) return 0;
        if (!load_from_file(M, path)) return 0;
    }
    return 1;
}

static void show_result(const Matrix *C, double secs)
{
    printf("\nResultado: matriz %zu x %zu   (calculo: %.6f s)\n", C->r, C->c, secs);

    if (C->r * C->c <= PRINT_LIMIT) {
        print_matrix(stdout, C);
        return;
    }

    size_t sr = C->r < 5 ? C->r : 5, sc = C->c < 5 ? C->c : 5;
    printf("Canto superior esquerdo (%zux%zu):\n", sr, sc);
    for (size_t i = 0; i < sr; i++) {
        for (size_t j = 0; j < sc; j++) printf("%14.6g ", AT(C, i, j));
        printf("\n");
    }
    const char *out = "resultado.txt";
    FILE *f = fopen(out, "wb");
    if (!f) { perror("fopen"); return; }
    setvbuf(f, NULL, _IOFBF, 1 << 20);
    double t0 = agora();
    print_matrix(f, C);
    fclose(f);
    printf("Matriz completa guardada em %s (escrita: %.3f s)\n", out, agora() - t0);
}

static double max_rel_err(const Matrix *X, const Matrix *Y)
{
    double worst = 0;
    for (size_t i = 0; i < X->r * X->c; i++) {
        double d = fabs(X->d[i] - Y->d[i]);
        double s = fabs(Y->d[i]) + 1.0;
        if (d / s > worst) worst = d / s;
    }
    return worst;
}

static int run_tests(void)
{
    const size_t dims[][3] = {
        {1,1,1}, {3,5,2}, {4,4,4}, {7,13,9}, {17,300,5}, {64,64,64},
        {101,257,513}, {5,1,700}, {130,520,131}, {1,1000,1}
    };
    int fails = 0;
    rng_state = 88172645463325252ULL;
    for (size_t t = 0; t < sizeof dims / sizeof dims[0]; t++) {
        Matrix A, B, C, R;
        mat_alloc(&A, dims[t][0], dims[t][1]);
        mat_alloc(&B, dims[t][1], dims[t][2]);
        for (size_t i = 0; i < A.r * A.c; i++) A.d[i] = rand_range(-5, 5);
        for (size_t i = 0; i < B.r * B.c; i++) B.d[i] = rand_range(-5, 5);
        mat_mul(&A, &B, &C);
        mat_alloc(&R, A.r, B.c);
        mat_mul_naive(&A, &B, &R);
        double e = max_rel_err(&C, &R);
        printf("%4zu x %-4zu * %4zu x %-4zu  erro relativo max = %.2e  %s\n",
               A.r, A.c, B.r, B.c, e, e < 1e-10 ? "OK" : "FALHOU");
        if (e >= 1e-10) fails++;
        mat_free(&A); mat_free(&B); mat_free(&C); mat_free(&R);
    }

    Matrix A, B, H, K;
    mat_alloc(&A, 2, 2); mat_alloc(&B, 2, 2);
    double av[] = {1, 2, 3, 4}, bv[] = {0, 5, 6, 7};
    memcpy(A.d, av, sizeof av); memcpy(B.d, bv, sizeof bv);
    mat_hadamard(&A, &B, &H);
    mat_kronecker(&A, &B, &K);
    int okh = H.d[0] == 0 && H.d[1] == 10 && H.d[2] == 18 && H.d[3] == 28;
    int okk = K.r == 4 && K.c == 4 && K.d[1] == 5 && K.d[2] == 0 && K.d[3] == 10 &&
              K.d[15] == 28 && K.d[8] == 0 && K.d[9] == 15;
    printf("Hadamard 2x2: %s | Kronecker 2x2: %s\n", okh ? "OK" : "FALHOU", okk ? "OK" : "FALHOU");
    fails += !okh + !okk;
    mat_free(&A); mat_free(&B); mat_free(&H); mat_free(&K);
    return fails;
}

static void run_bench(size_t n)
{
    Matrix A, B, C;
    rng_state = 88172645463325252ULL;
    if (!mat_alloc(&A, n, n) || !mat_alloc(&B, n, n)) { fprintf(stderr, "Sem memoria\n"); return; }
    for (size_t i = 0; i < n * n; i++) { A.d[i] = rand_range(-1, 1); B.d[i] = rand_range(-1, 1); }
    double t0 = agora();
    mat_mul(&A, &B, &C);
    double dt = agora() - t0;
    printf("%zux%zu * %zux%zu: %.3f s  (%.2f GFLOPS)\n", n, n, n, n, dt,
           2.0 * n * n * n / dt / 1e9);
    mat_free(&A); mat_free(&B); mat_free(&C);
}

int main(int argc, char **argv)
{
    if (argc > 1 && strcmp(argv[1], "--teste") == 0) return run_tests() ? 1 : 0;
    if (argc > 2 && strcmp(argv[1], "--bench") == 0) { run_bench((size_t)atol(argv[2])); return 0; }

    rng_state = (uint64_t)time(NULL) * 2654435761ULL + 88172645463325252ULL;
    setvbuf(stdin, NULL, _IOFBF, 1 << 20); 

    long op;
    printf("Operacao entre matrizes:\n"
           "  1) Produto matricial   A x B\n"
           "  2) Produto de Hadamard A .* B (elemento a elemento)\n"
           "  3) Produto de Kronecker A (x) B\n");
    if (!ask_long("Escolha: ", 1, 3, &op)) return 1;

    Matrix A = {0}, B = {0}, C = {0};
    if (!read_matrix(&A, "A", 0, 0)) return 1;

    int ok;
    if (op == 1)      ok = read_matrix(&B, "B", (long)A.c, 0);
    else if (op == 2) ok = read_matrix(&B, "B", (long)A.r, (long)A.c);
    else              ok = read_matrix(&B, "B", 0, 0);
    if (!ok) return 1;

    double t0 = agora();
    if (op == 1)      ok = mat_mul(&A, &B, &C);
    else if (op == 2) ok = mat_hadamard(&A, &B, &C);
    else              ok = mat_kronecker(&A, &B, &C);
    double dt = agora() - t0;

    if (!ok) { fprintf(stderr, "Erro: sem memoria ou resultado demasiado grande.\n"); return 1; }
    show_result(&C, dt);

    mat_free(&A); mat_free(&B); mat_free(&C);
    return 0;
}