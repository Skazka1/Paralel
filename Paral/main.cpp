#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <omp.h>

#define N 20000

int a[N][N];

int main() {
#if defined(_OPENMP)
    printf("Hello, OpenMP v.%d!\n", _OPENMP);
#else
    printf("OpenMP not supported!\n");
    return -1;
#endif

    for (int num_threads = 1; num_threads < 13; num_threads++) {
        omp_set_num_threads(num_threads);
        printf("=============%d threads=============\n", num_threads);

        double start, end, t;

        //заполнение нулями по строкам
        start = omp_get_wtime();
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                a[i][j] = 0;
            }
        }
        end = omp_get_wtime();
        t = end - start;
        printf("null time (raw): %.2lg seconds\n", t);

        //заполнение нулями по столбцам
        start = omp_get_wtime();
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                a[j][i] = 0;
            }
        }
        end = omp_get_wtime();
        t = end - start;
        printf("null time (column): %.2lg seconds\n", t);

        //заполнение по возрастанию по строкам
        start = omp_get_wtime();
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            int k = i * N;
            for (int j = 0; j < N; j++) {
                a[i][j] = k++;
            }
        }
        end = omp_get_wtime();
        t = end - start;
        printf("sequence time (raw): %.2lg seconds\n", t);

        //заполнение по возрастанию по столбцам
        start = omp_get_wtime();
#pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            int k = i;
            for (int j = 0; j < N; j++) {
                a[j][i] = k;
                k += N;
            }
        }
        end = omp_get_wtime();
        t = end - start;
        printf("sequence time (column): %.2lg seconds\n", t);

        printf("\n");
    }

    return 0;
}