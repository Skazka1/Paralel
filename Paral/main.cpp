#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <omp.h>

#define N 22000
#define L 12

#define SCHD_OPT static, 50
//#define SCHD_OPT dynamic, 50
//#define SCHD_OPT guided, 100

int a[N][N];

int main()
{
    int i, j, k = 1;
    int quantity_par[L] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    int threads;

    double time;
    double end_time;
    time = omp_get_wtime();
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            a[i][j] = 0;
    end_time = omp_get_wtime();
    printf("Zapolnenie nulyami dlya vydeleniya fizicheskoy pamyati: %lf\n", end_time - time);

    printf("\n");

    printf("------------------------------------------------\n");
    for (int y = 0; y < L; y++) {
        threads = quantity_par[y];
        omp_set_num_threads(threads);

        time = omp_get_wtime();
#pragma omp parallel for schedule(SCHD_OPT)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                a[i][j] = rand();
        end_time = omp_get_wtime();

        printf("Potokov: %2d | Vremya: %f sec\n", threads, end_time - time);
    }
    printf("\n");

    printf("-------------------------------------------------\n");
    for (int y = 0; y < L; y++) {
        threads = quantity_par[y];
        omp_set_num_threads(threads);

        time = omp_get_wtime();
#pragma omp parallel
        for (int i = 0; i < N; i++)
#pragma omp for schedule(SCHD_OPT)
            for (int j = 0; j < N; j++)
                a[i][j] = rand();
        end_time = omp_get_wtime();

        printf("Potokov: %2d | Vremya: %f sec\n", threads, end_time - time);
    }
    printf("\n");

    // Collapse
    printf("COLLAPSE(2) (obedinenie ciklov):\n");
    printf("-------------------------------\n");
    for (int y = 0; y < L; y++) {
        threads = quantity_par[y];
        omp_set_num_threads(threads);

        time = omp_get_wtime();
#pragma omp parallel for collapse(2) schedule(SCHD_OPT)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                a[i][j] = rand();
        end_time = omp_get_wtime();

        printf("Potokov: %2d | Vremya: %f sec\n", threads, end_time - time);
    }

    return 0;
}