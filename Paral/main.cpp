#include <iostream>
#include <cstdlib>
#include "mpi.h"
#include <cmath>
#include <iomanip>
#include <ctime>

using namespace std;

// Функция выделения памяти под вектор
double* alloc_array(int n)
{
    double* a = new double[n];
    return a;
}

// Функция освобождения памяти 
int free_array(double* a, int n)
{
    delete[] a;
    return 0;
}

// Параллельная функция умножения матрицы на вектор
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

    // Распределяем части матрицы по процессам
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

    // Умножение матрицы на вектор 100 раз
    for (int k = 0; k < 100; k++)
    {
        // Рассылаем вектор x всем процессам
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

        // Собираем результаты на процессе 0
        MPI_Gatherv(local_y, local_n, MPI_DOUBLE,
            y, recvcounts, recvdispls, MPI_DOUBLE,
            0, MPI_COMM_WORLD);

        // На процессе 0 обновляем вектор x для следующей итерации
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

// Последовательная версия для сравнения
int mult_mv_sequential(int n, double* A, double* x, double* y)
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
    cout << "Sequential execution time: " 
        << (end_time - start_time) << " seconds" << endl;

    return 0;
}

// Функция для сравнения результатов
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
    // Инициализация MPI
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Определение размера задачи
    int n = 840;  // размер кратен 2, 3, 4, 5, 6, 7, 8 для упрощения разбиения

    // Вычисляем размер подзадачи для каждого процесса
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
    double* A = NULL;  // матрица как одномерный массив (только на процессе 0)
    double* x = alloc_array(n);
    double* y = alloc_array(n);
    double* x_sequential = alloc_array(n);
    double* y_sequential = alloc_array(n);

    // Инициализация массивов
    for (int i = 0; i < n; i++) {
        x[i] = 0;
        y[i] = 0;
        x_sequential[i] = 0;
        y_sequential[i] = 0;
    }

    // Заполнение матрицы А и вектора х (только на процессе 0)
    if (rank == 0)
    {
        A = alloc_array(n * n);

        // Инициализация генератора случайных чисел
        srand(time(NULL));

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                A[i * n + j] = (double)rand() / RAND_MAX;
            }
            x[i] = (double)rand() / RAND_MAX;
            y[i] = 0;

            // Копируем данные для последовательной версии
            x_sequential[i] = x[i];
            y_sequential[i] = 0;
        }

        cout << "=== SEQUENTIAL VERSION ===" << endl;
        // Выполнение последовательной версии
        mult_mv_sequential(n, A, x_sequential, y_sequential);
        cout << endl;

        cout << "=== PARALLEL VERSION ===" << endl;
    }

    // Синхронизация перед параллельными вычислениями
    MPI_Barrier(MPI_COMM_WORLD);

    // Замер времени начала параллельных вычислений
    double start_time = MPI_Wtime();

    // Выполнение параллельной версии
    mult_mv(n, A, x, y, rank, size);

    // Замер времени окончания параллельных вычислений
    double end_time = MPI_Wtime();

    // Вывод времени выполнения на процессе 0
    if (rank == 0) {
        cout << "Parallel execution time: "
            << (end_time - start_time) << " seconds" << endl;

        cout << endl << "=== RESULTS COMPARISON ===" << endl;

        // Сравнение результатов
        if (compare_results(y_sequential, y, n)) {
            cout << "Results match with high precision!" << endl;
        }
        else {
            cout << "Results differ!" << endl;
        }

        // Вывод первых нескольких элементов для проверки
        cout << endl << "First 10 elements of result vector:" << endl;
        cout << "Sequential: ";
        for (int i = 0; i < min(10, n); i++) {
            cout << y_sequential[i] << " ";
        }
        cout << endl << "Parallel:   ";
        for (int i = 0; i < min(10, n); i++) {
            cout << y[i] << " ";
        }
        cout << endl;

        cout << endl << "The Program is RUN on " << size << " CPU(s)" << endl;
        cout << "Final y[0] = " << y[0] << endl;

        // Освобождаем память матрицы A
        free_array(A, n * n);
    }

    // Освобождение памяти
    free_array(x, n);
    free_array(y, n);
    free_array(x_sequential, n);
    free_array(y_sequential, n);

    MPI_Finalize();
    return 0;
}