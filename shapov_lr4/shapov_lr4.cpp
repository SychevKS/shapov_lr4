#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <chrono>
#include <omp.h>

void gaussian_elimination(std::vector<double>& a, std::vector<double>& b, std::vector<double>& x, int n)
{
    // Forward elimination
    for (int k = 0; k < n - 1; k++) {
        double pivot = a[k * n + k];
        #pragma omp parallel for shared(a, b)
        for (int i = k + 1; i < n; i++) {
            double lik = a[i * n + k] / pivot;
            for (int j = k; j < n; j++) {
                a[i * n + j] -= lik * a[k * n + j];
            }
            b[i] -= lik * b[k];
        }
    }

    // Backward substitution
    for (int k = n - 1; k >= 0; k--) {
        x[k] = b[k];
        for (int i = k + 1; i < n; i++) {
            x[k] -= a[k * n + i] * x[i];
        }
        x[k] /= a[k * n + k];
    }

    // Print the solution vector
    std::cout << "Solution vector:\n";
    for (int i = 0; i < n; i++) {
        std::cout << "x[" << i << "] = " << x[i] << "\n";
    }
}

int main()
{
    srand(time(0));
    int n = 10;

    std::vector<double> a(n*n);
    std::vector<double> b(n);
    std::vector<double> x(n);

    for (int i = 0; i < n*n; ++i) {
        a[i] = rand() % 100 + 1;
    }
    for (int i = 0; i < n; ++i) {
        b[i] = rand() % 100 + 1;
    }

    // solve the system using Gaussian elimination with forward and backward algorithms
    gaussian_elimination(a, b, x, n);

    return 0;
}