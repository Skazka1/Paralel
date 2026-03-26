#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    const int N = 23; // Длина векторов (может быть не кратна количеству процессов)

    // Определяем размер локальной части для каждого процесса
    int local_size = N / world_size;
    int remainder = N % world_size;
    int my_local_size = local_size + (world_rank < remainder ? 1 : 0);

    // Векторы для хранения локальных частей
    std::vector<int> local_vec1(my_local_size);
    std::vector<int> local_vec2(my_local_size);

    // Буферы для рассылки (используются только процессом 0)
    std::vector<int> send_vec1, send_vec2;
    std::vector<int> send_counts(world_size), displs(world_size);

    if (world_rank == 0) {
        std::vector<int> full_vec1(N);
        std::vector<int> full_vec2(N);

        // Заполнение первого вектора последовательными значениями 1, 2, 3, ...
        for (int i = 0; i < N; ++i) {
            full_vec1[i] = i + 1;
        }

        // Заполнение второго вектора случайными значениями от 1 до 100
        std::srand(static_cast<unsigned>(std::time(nullptr)));
        for (int i = 0; i < N; ++i) {
            full_vec2[i] = 1 + std::rand() % 100;
        }

        // Подготовка массивов
        int offset = 0;
        for (int i = 0; i < world_size; ++i) {
            int size_i = local_size + (i < remainder ? 1 : 0);
            send_counts[i] = size_i;
            displs[i] = offset;
            offset += size_i;
        }


        send_vec1 = full_vec1;
        send_vec2 = full_vec2;
    }

    // Рассылка первого вектора частями
    MPI_Scatterv(
        send_vec1.data(), send_counts.data(), displs.data(), MPI_INT,
        local_vec1.data(), my_local_size, MPI_INT,
        0, MPI_COMM_WORLD
    );

    // Рассылка второго вектора частями
    MPI_Scatterv(
        send_vec2.data(), send_counts.data(), displs.data(), MPI_INT,
        local_vec2.data(), my_local_size, MPI_INT,
        0, MPI_COMM_WORLD
    );

    // Вычисление локальной части скалярного произведения
    long long local_dot = 0;
    for (int i = 0; i < my_local_size; ++i) {
        local_dot += static_cast<long long>(local_vec1[i]) * local_vec2[i];
    }

    // Сбор локальных результатов на процессе 0
    long long global_dot = 0;
    MPI_Reduce(&local_dot, &global_dot, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // Вывод результата процессом 0
    if (world_rank == 0) {
        std::cout << "Dlinna vectorov: " << N << std::endl;
        std::cout << "kol-vo processov: " << world_size << std::endl;
        std::cout << "skalyarnoe proizvedenie: " << global_dot << std::endl;

        // Дополнительная проверка: вычисление полного произведения для верификации
        long long check_dot = 0;
        std::vector<int> full_vec1(N), full_vec2(N);
        for (int i = 0; i < N; ++i) full_vec1[i] = i + 1;
        std::srand(static_cast<unsigned>(std::time(nullptr))); // В реальности нужно сохранить seed
        for (int i = 0; i < N; ++i) full_vec2[i] = 1 + std::rand() % 100;
        for (int i = 0; i < N; ++i) check_dot += static_cast<long long>(full_vec1[i]) * full_vec2[i];
        std::cout << "Proverka: " << check_dot << std::endl;
    }

    MPI_Finalize();
    return 0;
}