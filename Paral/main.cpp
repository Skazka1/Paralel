#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include "locale.h"

using namespace std;


int Calc_ser(double** u, double** f, int N, double eps);  // последовательная
int Calc_blk(double** u, double** f, int N, double eps);  // блочная последовательная
int Calc_par(double** u, double** f, int N, double eps);  // параллельная (блочная)
int Calc_par_tasks(double** u, double** f, int N, double eps);  // параллельная (блочная) с задачами


void Init(double** u, double** f, int N);
double** new_arr(int N);
void delete_arr(double** arr, int N);
void Output(double** u, int N);


bool CompareResults(double** u1, double** u2, int N, double tolerance = 1e-8);


int main(int argc, char** argv)
{
    double** u = NULL, ** f = NULL;

    const int N = 5000;        // Количество точек сетки по каждой размерности
    const double eps = 0.001;   // Точность вычислений
    int icnt;                    // Количество итераций
    double stime;                // Время решения
    double start_time, end_time;

    f = new_arr(N);              // Выделение памяти под правую часть значений уравнения
    u = new_arr(N + 2);          // Выделение памяти под неизвестные и краевые условия


    double** u_ser = new_arr(N + 2);
    double** u_blk = new_arr(N + 2);
    double** u_par = new_arr(N + 2);
    double** u_par_tasks = new_arr(N + 2);

    //  Последовательная реализация
    cout << "\n\t*** Posledovatelnaya versiya ***\n";
    Init(u_ser, f, N);                 
    start_time = omp_get_wtime();
    icnt = Calc_ser(u_ser, f, N, eps);  
    end_time = omp_get_wtime();
    stime = end_time - start_time;
    cout << "Vremya resheniya = " << stime << " sec" << endl;
    cout << "Kolichestvo iteratsiy = " << icnt << endl;
    cout << "Rezultaty:\n";
    Output(u_ser, N);                  
    cout << "\n\t*** Blochnaya posledovatelnaya versiya ***\n";
    Init(u_blk, f, N);                  
    start_time = omp_get_wtime();
    icnt = Calc_blk(u_blk, f, N, eps); 
    end_time = omp_get_wtime();
    stime = end_time - start_time;
    cout << "Vremya resheniya = " << stime << " sec" << endl;
    cout << "Kolichestvo iteratsiy = " << icnt << endl;
    cout << "Rezultaty:\n";
    Output(u_blk, N);


    if (CompareResults(u_ser, u_blk, N)) {
        cout << "+++ OK! Blochnye rezultaty sovpadayut s posledovatelnymi +++" << endl;
    }
    else {
        cout << "!!! OSHIBKA !!! Blochnye rezultaty NE sovpadayut s posledovatelnymi !!!" << endl;
    }

  
    cout << "\n\t*** Parallelnaya versiya (parallel for) ***\n";
    Init(u_par, f, N);                  
    start_time = omp_get_wtime();
    icnt = Calc_par(u_par, f, N, eps); 
    end_time = omp_get_wtime();
    stime = end_time - start_time;
    cout << "Vremya resheniya = " << stime << " sec" << endl;
    cout << "Kolichestvo iteratsiy = " << icnt << endl;
    cout << "Rezultaty:\n";
    Output(u_par, N);

   
    if (CompareResults(u_blk, u_par, N)) {
        cout << "+++ OK! Parallelnye rezultaty (for) sovpadayut s blochnymi posledovatelnymi +++" << endl;
    }
    else {
        cout << "!!! OSHIBKA !!! Parallelnye rezultaty (for) NE sovpadayut s blochnymi posledovatelnymi !!!" << endl;
    }

  
    cout << "\n\t*** Parallelnaya versiya (omp task) ***\n";
    Init(u_par_tasks, f, N);           
    start_time = omp_get_wtime();
    icnt = Calc_par_tasks(u_par_tasks, f, N, eps); 
    end_time = omp_get_wtime();
    stime = end_time - start_time;
    cout << "Vremya resheniya = " << stime << " sec" << endl;
    cout << "Kolichestvo iteratsiy = " << icnt << endl;
    cout << "Rezultaty:\n";
    Output(u_par_tasks, N);


    if (CompareResults(u_blk, u_par_tasks, N)) {
        cout << "+++ OK! Parallelnye rezultaty (task) sovpadayut s blochnymi posledovatelnymi +++" << endl;
    }
    else {
        cout << "!!! OSHIBKA !!! Parallelnye rezultaty (task) NE sovpadayut s blochnymi posledovatelnymi !!!" << endl;
    }

    // Освобождение памяти массивов
    cout << "\nOsvobozhdenie pamyati zaversheno." << endl;

    delete_arr(f, N);
    delete_arr(u_ser, N + 2);
    delete_arr(u_blk, N + 2);
    delete_arr(u_par, N + 2);
    delete_arr(u_par_tasks, N + 2);
    delete_arr(u, N + 2);  // исходный массив u не использовался

    return 0;
}


int Calc_ser(double** u, double** f, int N, double eps)
{
    double max;                // Максимальная ошибка на итерации
    double h = 1.0 / (N + 1);  // Величина шага
    int icnt = 0;              // Количество итераций

    do
    {
        icnt++;
        max = 0;
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++)
            {
                double u0 = u[i][j];
                u[i][j] = 0.25 * (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1] - h * h * f[i - 1][j - 1]);
                double d = fabs(u[i][j] - u0);      // Разность нового значения неизвестной и значения с предыдущей итерации
                if (d > max)                         // Поиск максимальной ошибки
                    max = d;
            }
    } while (max > eps);

    return icnt;
}

// Последовательная функция, реализующая блочный алгоритм Гаусса-Зейделя
int Calc_blk(double** u, double** f, int N, double eps)
{
    double max;
    double h = 1.0 / (N + 1);
    int icnt = 0;

    const int BlockSize = 40;  // Размер блока
    int bcnt;                  // Количество блоков в ряд

    if (N % BlockSize == 0) // Если количество точек по каждому из направлений сетки делится нацело на размер блока, то проводятся вычисления
    {
        bcnt = N / BlockSize;
        do
        {
            icnt++;
            max = 0;

            // Волновая схема обхода блоков
            // Всего диагоналей: 2*bcnt - 1
            for (int diag = 0; diag < 2 * bcnt - 1; diag++)
            {
                // Перебираем все возможные индексы блоков i_block
                for (int i_block = 0; i_block < bcnt; i_block++)
                {
                    int j_block = diag - i_block;  // j_block определяется как diag - i_block

                    // Проверяем, что j_block находится в допустимых пределах [0, bcnt-1]
                    if (j_block >= 0 && j_block < bcnt)
                    {
                        // Обрабатываем блок (i_block, j_block)
                        // Границы блока в терминах индексов сетки
                        int i_start = i_block * BlockSize + 1;
                        int i_end = (i_block + 1) * BlockSize;
                        int j_start = j_block * BlockSize + 1;
                        int j_end = (j_block + 1) * BlockSize;

                        for (int i = i_start; i <= i_end; i++)
                        {
                            for (int j = j_start; j <= j_end; j++)
                            {
                                double old = u[i][j];
                                u[i][j] = 0.25 * (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1] - h * h * f[i - 1][j - 1]);
                                double diff = fabs(u[i][j] - old);
                                if (diff > max) max = diff;
                            }
                        }
                    }
                }
            }
        } while (max > eps);
    }
    else
    {
        cout << "OSHI BKA! N ne delitsya na BlockSize!" << endl;
        exit(1);
    }

    return icnt;
}

// Параллельная реализация блочного алгоритма Гаусса-Зейделя
int Calc_par(double** u, double** f, int N, double eps)
{
    double max;
    double h = 1.0 / (N + 1);
    int icnt = 0;

    const int BlockSize = 40;  // Размер блока
    int bcnt;                  // Количество блоков в ряд

    if (N % BlockSize == 0) 
    {
        bcnt = N / BlockSize;

        do
        {
            icnt++;
            max = 0;

            for (int diag = 0; diag < 2 * bcnt - 1; diag++)
            {
#pragma omp parallel for reduction(max:max) schedule(dynamic)
                for (int i_block = 0; i_block < bcnt; i_block++)
                {
                    int j_block = diag - i_block;

                    // Проверяем, что j_block в допустимых пределах
                    if (j_block >= 0 && j_block < bcnt)
                    {
                        int i_start = i_block * BlockSize + 1;
                        int i_end = (i_block + 1) * BlockSize;
                        int j_start = j_block * BlockSize + 1;
                        int j_end = (j_block + 1) * BlockSize;

                        for (int i = i_start; i <= i_end; i++)
                        {
                            for (int j = j_start; j <= j_end; j++)
                            {
                                double old = u[i][j];
                                u[i][j] = 0.25 * (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1] - h * h * f[i - 1][j - 1]);
                                double diff = fabs(u[i][j] - old);
                                if (diff > max) max = diff;
                            }
                        }
                    }
                }
            }
        } while (max > eps);
    }
    else
    {
        cout << "OSHI BKA! N ne delitsya na BlockSize!" << endl;
        exit(1);
    }

    return icnt;
}


int Calc_par_tasks(double** u, double** f, int N, double eps)
{
    double max;
    double h = 1.0 / (N + 1);
    int icnt = 0;

    const int BlockSize = 40;  // Размер блока
    int bcnt;                  // Количество блоков в ряд

    if (N % BlockSize == 0)
    {
        bcnt = N / BlockSize;

        do
        {
            icnt++;
            max = 0;

            // Волновая схема обхода блоков
            for (int diag = 0; diag < 2 * bcnt - 1; diag++)
            {
                double* local_maxs = new double[bcnt];
                int* block_indices = new int[bcnt];
                int blocks_count = 0;

                for (int i_block = 0; i_block < bcnt; i_block++)
                {
                    int j_block = diag - i_block;
                    if (j_block >= 0 && j_block < bcnt)
                    {
                        block_indices[blocks_count] = i_block;
                        local_maxs[blocks_count] = 0.0;
                        blocks_count++;
                    }
                }

                // Параллельное выполнение задач для блоков на диагонали
#pragma omp parallel
                {
#pragma omp single
                    {
                        for (int idx = 0; idx < blocks_count; idx++)
                        {
                            int i_block = block_indices[idx];
                            int j_block = diag - i_block;

                            int i_start = i_block * BlockSize + 1;
                            int i_end = (i_block + 1) * BlockSize;
                            int j_start = j_block * BlockSize + 1;
                            int j_end = (j_block + 1) * BlockSize;

                            
#pragma omp task firstprivate(i_start, i_end, j_start, j_end, idx) shared(local_maxs, u, f, h)
                            {
                                double local_max = 0.0;
                                for (int i = i_start; i <= i_end; i++)
                                {
                                    for (int j = j_start; j <= j_end; j++)
                                    {
                                        double old = u[i][j];
                                        u[i][j] = 0.25 * (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1] - h * h * f[i - 1][j - 1]);
                                        double diff = fabs(u[i][j] - old);
                                        if (diff > local_max) local_max = diff;
                                    }
                                }
                                local_maxs[idx] = local_max;
                            }
                        }
                    }
                }

                // Находим максимальную ошибку среди всех блоков на диагонали
                for (int idx = 0; idx < blocks_count; idx++)
                {
                    if (local_maxs[idx] > max) max = local_maxs[idx];
                }

                delete[] local_maxs;
                delete[] block_indices;
            }
        } while (max > eps);
    }
    else
    {
        cout << "OSHI BKA! N ne delitsya na BlockSize!" << endl;
        exit(1);
    }

    return icnt;
}


double** new_arr(int N)
{
    double** f = new double* [N];
    for (int i = 0; i < N; i++)
    {
        f[i] = new double[N];
        // Инициализируем нулями для избежания мусора
        for (int j = 0; j < N; j++)
            f[i][j] = 0.0;
    }
    return f;
}

// Функция освобождения памяти 2D массива
void delete_arr(double** arr, int N)
{
    for (int i = 0; i < N; i++)
    {
        delete[] arr[i];
    }
    delete[] arr;
}

// Задание граничных значений
double G(double x, double y)
{
    if (x == 0) return 1 - 2 * y;
    if (x == 1) return -1 + 2 * y;
    if (y == 0) return 1 - 2 * x;
    if (y == 1) return -1 + 2 * x;
    return 0.0; // внутренние точки не используются для граничных условий
}

// Задание правой части
double F(double x, double y)
{
    return 2.2;
}

// Инициализация массивов правой части и краевых условий
void Init(double** u, double** f, int N)
{
    double h = 1.0 / (N + 1);

    // Инициализация правой части
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            f[i][j] = F((i + 1) * h, (j + 1) * h);
    }

    // Инициализация внутренних точек и границ по x
    for (int i = 1; i <= N; i++)
    {
        for (int j = 1; j <= N; j++)
            u[i][j] = 0.2;  // начальное приближение

        // Левая и правая границы (y = 0 и y = 1)
        u[i][0] = G(i * h, 0);
        u[i][N + 1] = G(i * h, (N + 1) * h);
    }

    // Инициализация нижней и верхней границ (x = 0 и x = 1)
    for (int j = 0; j <= N + 1; j++)
    {
        u[0][j] = G(0, j * h);
        u[N + 1][j] = G((N + 1) * h, j * h);
    }
}

// Функция вывода прореженной матрицы решения
void Output(double** u, int N)
{
    const int K = 5;
    cout << fixed << setprecision(8);
    for (int i = 0; i <= K; i++)
    {
        for (int j = 0; j <= K; j++)
            cout << setw(12) << u[i * (N + 1) / K][j * (N + 1) / K];
        cout << endl;
    }
}

// Функция сравнения результатов двух массивов
bool CompareResults(double** u1, double** u2, int N, double tolerance)
{
    double max_diff = 0.0;
    cout << "Proveryaem rezultaty..." << endl;

    for (int i = 1; i <= N; i++)
    {
        for (int j = 1; j <= N; j++)
        {
            double diff = fabs(u1[i][j] - u2[i][j]);
            if (diff > max_diff) max_diff = diff;
            if (diff > tolerance)
            {
                cout << "Bolshaya raznitsa v tochke (" << i << "," << j << "): "
                    << u1[i][j] << " vs " << u2[i][j] << ", diff = " << diff << endl;
                cout << "Dopusk = " << tolerance << endl;
                return false;
            }
        }
    }
    cout << "Maksimalnaya raznitsa mezhdu massivami = " << max_diff << endl;
    cout << "Dopusk = " << tolerance << endl;
    return max_diff <= tolerance;
}