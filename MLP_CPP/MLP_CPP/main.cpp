#include "matrix.h"
#include "mlp.h"
#include "nn.h"

void test_lib()
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
    //std::vector<double> data{ 1,2,3,4,5,6,7,8,9 };
    //Matrix matrix1(data, 3, 3);
    //std::vector<double> data2{ 10,20,30,40,50,60,70,80,90 };
    //Matrix matrix2(data2, 3, 3);

    //// Test addition
    //Matrix resultAdd = matrix1 + matrix2;
    //resultAdd.print();

    //// Test subtraction
    //Matrix resultSub = matrix1 - matrix2;
    //resultSub.print();

    //// Test multiplication by scalar
    //Matrix resultMulScalar = matrix1 * 2;
    //resultMulScalar.print();

    //// Test multiplication by matrix
    //Matrix resultMulMatrix = matrix1 * matrix2;
    //resultMulMatrix.print();

    //// Test division by scalar
    //Matrix resultDivScalar = matrix1 / 2;
    //resultDivScalar.print();

    //// Test square
    //std::cout << "square\n";
    //Matrix::square(matrix1).print();
    //Matrix c = matrix1;
    //c.square().print();

    //// Test transpose
    //std::cout << "transpose\n";
    //std::vector<double> d3{ 1,2,3,4,5,6 };
    //Matrix::transpose(Matrix(d3, 2, 3)).print();
    //Matrix(d3, 2, 3).transpose().print();

    //// Test addition assignment
    //matrix1 += matrix2;
    //matrix1.print();

    //// Test subtraction assignment
    //matrix1 -= matrix2;
    //matrix1.print();

    //// Test multiplication assignment by scalar
    //matrix1 *= 2;
    //matrix1.print();

    //// Test multiplication assignment by matrix
    //matrix1 *= matrix2;
    //matrix1.print();

    //// Test division assignment by scalar
    //matrix1 /= 2;
    //matrix1.print();

    //Matrix mat3(3, 5);
    //mat3.randomize(-1, 1);
    //std::cout << mat3;

    //// Test max
    //Matrix aa(3, 10);
    //aa.randomize(0, 1);
    //std::cout << "original: " << aa;
    //std::cout << "max: " << aa.max();
    //Matrix m(aa.max());
    //std::cout << "subtrack: " << m.mismatch_dim_subtract(aa);

    //// Test softmax
    //Matrix matrix(3, 10);
    //matrix.randomize(0, 1);
    //std::cout << "original: " << matrix;
    //Matrix max = matrix.max();
    //std::cout << "max: " << max;
    //Matrix _V = Matrix::mismatch_dim_subtract(matrix, max);
    //std::cout << "_V: " << _V;
    //Matrix e_V = Matrix::apply_func(_V, [](double x) {return std::exp(x); });
    //std::cout << "e_V: " << e_V;
    //Matrix sum = e_V.sum(0);
    //std::cout << "sum: " << sum;
    //Matrix Z = Matrix::mismatch_dim_divide(e_V, sum);
    //std::cout << "Z: " << Z;

    //// Test cross-entropy loss
    //Matrix matrix(3, 10);
    //matrix.randomize(0, 1);
    //std::cout << "mat1: " << matrix;
    //Matrix matrix2(3, 10);
    //matrix2.randomize(0, 1);
    //std::cout << "mat2: " << matrix2;
    //Matrix log_m = Matrix::apply_func(matrix2, [](double x) {return std::log(x); });
    //std::cout << "log_m: " << log_m;
    //Matrix mul = Matrix::element_wise_mul(matrix, log_m);
    //std::cout << "mul: " << mul;
    //Matrix sum = mul.sum(-1);
    //std::cout << "sum: " << sum;
    //std::cout << "loss: " << (0. - sum) / matrix.get_width();

    //
	Matrix a(5, 3);
    a.randomize(0, 1);
    Matrix b(5, 2);
    b.randomize(0, 1);
    std::cout << a << b;
    //Matrix::shuffle(a, b);
    /*std::cout << a << b;

    std::cout << a.extract(2, 1);
    b.drop(1, 1);
    std::cout << b;
    std::cout << a.extract(1, 2, 1);*/

    std::cout << a.argmax(0);
}

void test_mlp()
{
    MLP mlp(16, 100, 26, .1, 100, 50, "relu");
    //mlp.test();
    Matrix X_train, y_train, X_test, y_test;
    X_train.load_data_txt(16000, 16, R"(../data/writing/X_train.txt)");
    y_train.load_data_txt(16000, 26, R"(../data/writing/y_train.txt)");
    X_test.load_data_txt(4000, 16, R"(../data/writing/X_test.txt)");
    y_test.load_data_txt(4000, 26, R"(../data/writing/y_test.txt)");

    mlp.fit(X_train, y_train);

    mlp.eval(X_test, y_test);

    mlp.save_model(R"(../model/writing/)");
}

void test_nn()
{
    NeuralNet net(std::vector<int>{16, 10, 15, 10, 26}, .1, 100, 50, "relu", "sgd");
    net.test();
}

int main()
{
    //test_lib();
    //test_mlp();
    test_nn();
	return 0;
}