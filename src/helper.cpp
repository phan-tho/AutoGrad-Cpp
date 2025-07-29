#include <cstdlib>
#include <iostream>

double** zeros(int rows, int cols) {
    double** res = new double*[rows];
    for (int i = 0; i < rows; i++)  *(res + i) = new double[cols];
    return res;
}

double** random(int rows, int cols) {
    double** res = zeros(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            res[i][j] = (rand() % 50)*1.0 / 10;
        }
    }

    return res;
}

void printArr(double** arr, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << arr[i][j] << ' ';
        }
        std::cout << "\n";
    }
}