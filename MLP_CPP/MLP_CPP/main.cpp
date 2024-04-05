#include "matrix.h"
int main()
{
	/*Matrix a(2, 2);
	Matrix b(2, 2);
	a += 2;
	b -= 1;
	std::cout << "a " << a;
	std::cout << "b " << b;
	std::cout << "a+b " << a + b;
	std::cout << "a-b " << a - b;
	std::cout << "a*b " << a * b;
	std::cout << "a/b " << a / b;
	a += b;
	std::cout << "a+=b" << a;
	a -= b;
	std::cout << "a-=b" << a;
	a *= b;
	std::cout << "a*=b" << a;
	a /= 2.0;
	std::cout << "a/=2" << a;*/
	std::vector<double> data{1,2,3,4,5,6,7,8,9};
	Matrix matrix1(data, 3, 3);
	std::vector<double> data2{ 10,20,30,40,50,60,70,80,90 };
	Matrix matrix2(data2, 3, 3);

    // Test addition
    Matrix resultAdd = matrix1 + matrix2;
    resultAdd.print();

    // Test subtraction
    Matrix resultSub = matrix1 - matrix2;
    resultSub.print();

    // Test multiplication by scalar
    Matrix resultMulScalar = matrix1 * 2;
    resultMulScalar.print();

    // Test multiplication by matrix
    Matrix resultMulMatrix = matrix1 * matrix2;
    resultMulMatrix.print();

    // Test division by scalar
    Matrix resultDivScalar = matrix1 / 2;
    resultDivScalar.print();

    // Test addition assignment
    matrix1 += matrix2;
    matrix1.print();

    // Test subtraction assignment
    matrix1 -= matrix2;
    matrix1.print();

    // Test multiplication assignment by scalar
    matrix1 *= 2;
    matrix1.print();

    // Test multiplication assignment by matrix
    matrix1 *= matrix2;
    matrix1.print();

    // Test division assignment by scalar
    matrix1 /= 2;
    matrix1.print();
	return 0;
}