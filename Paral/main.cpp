#include <vector>
#include <chrono>
#include <iostream>

#define N 20000

std::vector<std::vector<int>> a(N, std::vector<int>(N));

int main()
{
	int i, j;
	int k = 0;
	
	auto start = std::chrono::system_clock::now();

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			a[i][j] = 0;
		}
	}

	auto end = std::chrono::system_clock::now();

	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms - null" << std::endl;

	start = std::chrono::system_clock::now();

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			a[i][j] = k++;
		}
	}

	end = std::chrono::system_clock::now();

	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms - sequence" << std::endl;

	start = std::chrono::system_clock::now();

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			a[i][j] = rand();
		}
	}

	end = std::chrono::system_clock::now();

	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms - rand" << std::endl;

	start = std::chrono::system_clock::now();

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			a[j][i] = 0;
		}
	}

	end = std::chrono::system_clock::now();

	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms - null column" << std::endl;

	start = std::chrono::system_clock::now();
	k = 0;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			a[j][i] = k++;
		}
	}

	end = std::chrono::system_clock::now();

	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms - sequence column" << std::endl;

	start = std::chrono::system_clock::now();

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			a[j][i] = rand();
		}
	}

	end = std::chrono::system_clock::now();

	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms - rand column" << std::endl;
	
	return 0;
}