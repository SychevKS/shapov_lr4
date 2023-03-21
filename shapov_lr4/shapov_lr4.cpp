#include <iostream>
#include <vector>
#include <omp.h>

void gaussian_elimination(std::vector<double>& a, std::vector<double>& b, std::vector<double>& x, int n)
{
    // Forward elimination
    for (int k = 0; k < n - 1; k++) {
        double pivot = a[k * n + k];
        #pragma omp parallel for shared(a, b) num_threads(4)
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
    // create the matrix A and vector b
    std::vector<double> a = { 2, 1, -1, -3, -1, 2, -2, 1, 2 };
    std::vector<double> b = { 8, -11, -3 };

    int n = 3;
    std::vector<double> x(n);

    // solve the system using Gaussian elimination with forward and backward algorithms
    gaussian_elimination(a, b, x, n);

    return 0;
}