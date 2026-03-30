#include <iostream>
#include <cstdlib>
#include "mpi.h"
#include <cmath>
#include <iomanip>
#include <ctime>

using namespace std;

double* alloc_array(int n)
{
    double* a = new double[n];
    return a;
}

int free_array(double* a, int n)
{
    delete[] a;
    return 0;
}

int mult_mv(int n, double* A, double* x, double* y, int rank, int size)
{
    // Определение числа доступных процессов
    int n1 = n / size;
    int remainder = n % size;

    // Определяем количество строк для текущего процесса
    int local_n = n1;
    if (rank < remainder) {
        local_n++;
    }

    // Подготовка массивов для Scatterv и Gatherv
    int* sendcounts = nullptr;
    int* displs = nullptr;
    int* recvcounts = nullptr;
    int* recvdispls = nullptr;

    if (rank == 0) {
        sendcounts = new int[size];
        displs = new int[size];

        int current_offset = 0;
        for (int i = 0; i < size; i++) {
            int rows = n1;
            if (i < remainder) rows++;
            sendcounts[i] = rows * n;
            displs[i] = current_offset;
            current_offset += rows * n;
        }
    }

    // Выделяем память для локальной части матрицы
    double* local_A = new double[local_n * n];
    double* local_y = new double[local_n];

    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE,
        local_A, local_n * n, MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    // Подготовка массивов для Gatherv
    if (rank == 0) {
        recvcounts = new int[size];
        recvdispls = new int[size];

        int current_offset = 0;
        for (int i = 0; i < size; i++) {
            int rows = n1;
            if (i < remainder) rows++;
            recvcounts[i] = rows;
            recvdispls[i] = current_offset;
            current_offset += rows;
        }
    }

    for (int k = 0; k < 100; k++)
    {
        MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Вычисляем локальную часть результата
        for (int i = 0; i < local_n; i++)
        {
            local_y[i] = 0;
            for (int j = 0; j < n; j++)
            {
                local_y[i] += local_A[i * n + j] * x[j];
            }
        }

        MPI_Gatherv(local_y, local_n, MPI_DOUBLE,
            y, recvcounts, recvdispls, MPI_DOUBLE,
            0, MPI_COMM_WORLD);

        if (rank == 0) {
            for (int i = 0; i < n; i++) {
                x[i] = y[i];
            }
        }
    }

    // Освобождаем локальную память
    delete[] local_A;
    delete[] local_y;

    if (rank == 0) {
        delete[] sendcounts;
        delete[] displs;
        delete[] recvcounts;
        delete[] recvdispls;
    }

    return 0;
}

// Изменяем функцию, чтобы она возвращала время выполнения
double mult_mv_sequential(int n, double* A, double* x, double* y)
{
    double start_time = MPI_Wtime();

    for (int k = 0; k < 100; k++)
    {
        for (int i = 0; i < n; i++)
        {
            y[i] = 0;
            for (int j = 0; j < n; j++)
            {
                y[i] += A[i * n + j] * x[j];
            }
        }
        for (int i = 0; i < n; i++)
        {
            x[i] = y[i];
        }
    }

    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;
    cout << "Sequential execution time: "
        << elapsed << " seconds" << endl;
    return elapsed;
}

bool compare_results(double* y1, double* y2, int n, double epsilon = 1e-8)
{
    for (int i = 0; i < n; i++) {
        if (fabs(y1[i] - y2[i]) > epsilon) {
            cout << "Difference at index " << i << ": " << y1[i] << " vs " << y2[i] << endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 840;

    int n1 = n / size;
    int remainder = n % size;

    // Определяем локальный размер для текущего процесса


    int local_n = n1;
    if (rank < remainder) {
        local_n++;
    }

    if (rank == 0) {
        cout << "========================================" << endl;
        cout << "Matrix size: " << n << " x " << n << endl;
        cout << "Number of iterations: 100" << endl;
        cout << "Number of MPI processes: " << size << endl;
        cout << "Distribution: " << endl;
        for (int i = 0; i < size; i++) {
            int rows = n1;
            if (i < remainder) rows++;
            cout << "  Process " << i << ": " << rows << " rows" << endl;
        }
        cout << "========================================" << endl << endl;
    }

    // Выделение памяти 
    double* A = NULL;
    double* x = alloc_array(n);
    double* y = alloc_array(n);
    double* x_sequential = alloc_array(n);
    double* y_sequential = alloc_array(n);

    // Инициализация нулями
    for (int i = 0; i < n; i++) {
        x[i] = 0;
        y[i] = 0;
        x_sequential[i] = 0;
        y_sequential[i] = 0;
    }

    double sequential_time = 0.0;  // для хранения времени последовательной версии

    if (rank == 0)
    {
        A = alloc_array(n * n);

        srand(time(NULL));

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                A[i * n + j] = (double)rand() / RAND_MAX;
            }
            x[i] = (double)rand() / RAND_MAX;
            y[i] = 0;

            x_sequential[i] = x[i];
            y_sequential[i] = 0;
        }

        cout << "=== SEQUENTIAL VERSION ===" << endl;
        sequential_time = mult_mv_sequential(n, A, x_sequential, y_sequential);
        cout << endl;

        cout << "=== PARALLEL VERSION ===" << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    mult_mv(n, A, x, y, rank, size);
    double end_time = MPI_Wtime();

    if (rank == 0) {
        double parallel_time = end_time - start_time;
        cout << "Parallel execution time: "
            << parallel_time << " seconds" << endl;

        // Вычисляем ускорение и эффективность
        double speedup = sequential_time / parallel_time;
        double efficiency = speedup / size;

        cout << "Speedup: " << speedup << endl;
        cout << "Efficiency: " << efficiency << endl;

        cout << endl << "=== RESULTS COMPARISON ===" << endl;

        // Сравнение результатов
        if (compare_results(y_sequential, y, n)) {
            cout << "Results match with high precision!" << endl;
        }
        else {
            cout << "Results differ!" << endl;
        }

        cout << endl << "The Program is RUN on " << size << " CPU(s)" << endl;
        cout << "Final y[0] = " << y[0] << endl;

        free_array(A, n * n);
    }

    free_array(x, n);
    free_array(y, n);
    free_array(x_sequential, n);
    free_array(y_sequential, n);

    MPI_Finalize();
    return 0;
}