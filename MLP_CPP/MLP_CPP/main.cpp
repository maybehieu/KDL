#include "matrix.h"
#include "mlp.h"
#include "nn.h"
#include "svm.h"
#include "utils.h"

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
	/*Matrix a(5, 3);
    a.randomize(0, 1);
    Matrix b(5, 2);
    b.randomize(0, 1);
    std::cout << a << b;*/
    //Matrix::shuffle(a, b);
    /*std::cout << a << b;

    std::cout << a.extract(2, 1);
    b.drop(1, 1);
    std::cout << b;
    std::cout << a.extract(1, 2, 1);*/

    //std::cout << a.argmax(0);

	Matrix a(4, 5);
	a.randomize(0, 1, .01);
    std::cout << a;
    Matrix b(4, 1);
    b.concat(a, 2, 3, 1);
    std::cout << b;

}

void test_grad()
{
    //// error with normalization
    //Matrix X;
    //X.load_data_txt(2, 2, "../data/test/X.txt");
    //Matrix y;
    //y.load_data_txt(3, 2, "../data/test/Y.txt");
    //Matrix W;
    //W.load_data_txt(2, 3, "../data/test/W.txt");

    //// grad
    //Matrix A = softmax(W.transpose() * X);
    //Matrix E = A - y;
    //Matrix grad = X * Matrix::transpose(E);
    Matrix ss;
    ss.load_data_txt(3, 2, "../data/test/grad_test.txt");
    Matrix sm = softmax(ss);
    return;
}

std::vector<int> topo_str_to_vector(std::string s)
{
    s.erase(std::remove_if(s.begin(), s.end(), [](char c) { return c == '{' || c == '}'; }), s.end());
    std::vector<int> result;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        result.push_back(std::stoi(item));
    }
    return result;
}

void dataset_slow_load(Matrix& xtrain, Matrix& ytrain, Matrix& xtest, Matrix& ytest,
    const std::string& xtrain_path, const std::string& ytrain_path, const std::string& xtest_path, const std::string& ytest_path)
{
    xtrain.load_data_txt(xtrain_path);
    ytrain.load_data_txt(ytrain_path);
    xtest.load_data_txt(xtest_path);
    ytest.load_data_txt(ytest_path);
}

void dataset_quick_load(Matrix& xtrain, Matrix& ytrain, Matrix& xtest, Matrix& ytest,
    const std::string& xtrain_path, const std::string& ytrain_path, const std::string& xtest_path, const std::string& ytest_path)
{
    xtrain.load_data(xtrain_path);
    ytrain.load_data(ytrain_path);
    xtest.load_data(xtest_path);
    ytest.load_data(ytest_path);
}

void preview_dataset(const Matrix& xtrain, const Matrix& ytrain, const Matrix& xtest, const Matrix& ytest)
{
    xtrain.print_shape();
    ytrain.print_shape();
    xtest.print_shape();
    ytest.print_shape();
}

void test_mlp()
{
    MLP mlp(16, 100, 26, .1, 5000, 500, "relu");
    //mlp.test();
    Matrix X_train, y_train, X_test, y_test;
    X_train.load_data_txt(16000, 16, R"(../data/writing/X_train.txt)");
    y_train.load_data_txt(16000, 26, R"(../data/writing/y_train.txt)");
    X_test.load_data_txt(4000, 16, R"(../data/writing/X_test.txt)");
    y_test.load_data_txt(4000, 26, R"(../data/writing/y_test.txt)");

    mlp.fit(X_train, y_train);

    mlp.print_eval(X_test, y_test);

    //mlp.save_model(R"(../model/writing/)");
}

void test_nn()
{
    //NeuralNet net(std::vector<int>{16, 100, 150, 100, 26}, .001, 500, 1000, "relu", "mse", "sgd", "weight");
    //NeuralNet net(std::vector<int>{784, 150, 100, 2}, .0001, 20, 1000, "relu", "mse", "sgd", "xmnist_mse_sgd");
    //NeuralNet net(std::vector<int>{4, 15, 2}, .0007, 260, 50, "relu", "mse", "sgd", "iris_nonsep_mse_sgd");
    NeuralNet net(std::vector<int>{22, 150, 100, 2}, .0001, 100, 1000, "relu", "mse", "sgd", "xmushroom_mse_sgd");

    //net.simple_test();

	//net.test();

    std::cout << "Loading dataset, this might take a while...\n";
    Matrix X_train, y_train, X_test, y_test;
    // mnist
    /*dataset_quick_load(X_train, y_train, X_test, y_test, "F:\\Documents\\KDL\\MLP_CPP\\data\\mnist_xtrain.bin", "F:/Documents/KDL/MLP_CPP\\data\\mnist_ytrain.bin",
        "F:\\Documents\\KDL\\MLP_CPP\\data\\mnist_xtest.bin", "F:\\Documents\\KDL\\MLP_CPP\\data\\mnist_ytest.bin");*/
    /*dataset_quick_load(X_train, y_train, X_test, y_test, "F:\\Documents\\KDL\\MLP_CPP\\data\\01mnist_xtrain.bin", "F:/Documents/KDL/MLP_CPP\\data\\01mnist_ytrain.bin",
        "F:\\Documents\\KDL\\MLP_CPP\\data\\01mnist_xtest.bin", "F:\\Documents\\KDL\\MLP_CPP\\data\\01mnist_ytest.bin");*/
    // mushroom
    dataset_quick_load(X_train, y_train, X_test, y_test, "F:\\Documents\\KDL\\MLP_CPP\\data\\mushroom_xtrain.bin", "F:/Documents/KDL/MLP_CPP\\data\\mushroom_ytrain.bin",
        "F:\\Documents\\KDL\\MLP_CPP\\data\\mushroom_xtest.bin", "F:\\Documents\\KDL\\MLP_CPP\\data\\mushroom_ytest.bin");
    // iris
    /*dataset_quick_load(X_train, y_train, X_test, y_test, "F:\\Documents\\KDL\\MLP_CPP\\data\\iris_xtrain.bin", "F:/Documents/KDL/MLP_CPP\\data\\iris_ytrain.bin",
        "F:\\Documents\\KDL\\MLP_CPP\\data\\iris_xtest.bin", "F:\\Documents\\KDL\\MLP_CPP\\data\\iris_ytest.bin");*/
    /*dataset_quick_load(X_train, y_train, X_test, y_test, "F:\\Documents\\KDL\\MLP_CPP\\data\\iris_2_xtrain.bin", "F:/Documents/KDL/MLP_CPP\\data\\iris_2_ytrain.bin",
        "F:\\Documents\\KDL\\MLP_CPP\\data\\iris_2_xtest.bin", "F:\\Documents\\KDL\\MLP_CPP\\data\\iris_2_ytest.bin");*/

    /*X_train.load_data_txt(16000, 16, R"(../data/writing/X_train.txt)");
    y_train.load_data_txt(16000, 26, R"(../data/writing/y_train.txt)");
    X_test.load_data_txt(4000, 16, R"(../data/writing/X_test.txt)");
    y_test.load_data_txt(4000, 26, R"(../data/writing/y_test.txt)");*/
    std::cout << "Load dataset complete!\n";

    // load model
    if (false)
    {
	    net.load_model("../weights/best_mnist_mse_sgd.bin");
        std::cout << "Loaded pre-trained model!\n";
    }

    net.fit(X_train, y_train);

    net.print_eval(X_test, y_test);
}

void test_svm()
{
    SoftMarginSVM svm(1.0, 0.0001, 50, "mnist_0001_1");
    std::cout << "Loading training data...";

    Matrix X_train, y_train, X_test, y_test;
    // mnist
    dataset_quick_load(X_train, y_train, X_test, y_test, "F:\\Documents\\KDL\\MLP_CPP\\data\\mnist_xtrain.bin", "F:/Documents/KDL/MLP_CPP\\data\\mnist_ytrain.bin",
        "F:\\Documents\\KDL\\MLP_CPP\\data\\mnist_xtest.bin", "F:\\Documents\\KDL\\MLP_CPP\\data\\mnist_ytest.bin");
        // mushroom
    /*dataset_quick_load(X_train, y_train, X_test, y_test, "F:\\Documents\\KDL\\MLP_CPP\\data\\mushroom_xtrain.bin", "F:/Documents/KDL/MLP_CPP\\data\\mushroom_ytrain.bin",
        "F:\\Documents\\KDL\\MLP_CPP\\data\\mushroom_xtest.bin", "F:\\Documents\\KDL\\MLP_CPP\\data\\mushroom_ytest.bin");*/
    // iris
    /*dataset_quick_load(X_train, y_train, X_test, y_test, "F:\\Documents\\KDL\\MLP_CPP\\data\\iris_xtrain.bin", "F:/Documents/KDL/MLP_CPP\\data\\iris_ytrain.bin",
        "F:\\Documents\\KDL\\MLP_CPP\\data\\iris_xtest.bin", "F:\\Documents\\KDL\\MLP_CPP\\data\\iris_ytest.bin");*/
    /*dataset_quick_load(X_train, y_train, X_test, y_test, "F:\\Documents\\KDL\\MLP_CPP\\data\\iris_2_xtrain.bin", "F:/Documents/KDL/MLP_CPP\\data\\iris_2_ytrain.bin",
        "F:\\Documents\\KDL\\MLP_CPP\\data\\iris_2_xtest.bin", "F:\\Documents\\KDL\\MLP_CPP\\data\\iris_2_ytest.bin");*/

    // transform label from one-hot to single label for SVM
    y_train = y_train.argmax(0);
    y_test = y_test.argmax(0);

    std::cout << "\rFinished loading training data!\n";

    // load model
    if (true)
    {
	    svm.load_model(R"(F:\Documents\KDL\MLP_CPP\svm_weights\best_mnist_0001_1.bin)");
        std::cout << "Loaded pre-trained weight!\n";
    }

    //svm.fit(X_train, y_train);

    svm.print_eval(X_test, y_test);
}

void test_file()
{
    int64_t time = getTickcount();
    std::cout << "Loading dataset, this might take a while...\n";
    Matrix X_train, y_train, X_test, y_test;
    dataset_slow_load(X_train, y_train, X_test, y_test, "F:\\Documents\\KDL\\MLP_CPP\\data\\01mnist_xtrain.txt", "F:/Documents/KDL/MLP_CPP\\data\\01mnist_ytrain.txt",
        "F:\\Documents\\KDL\\MLP_CPP\\data\\01mnist_xtest.txt", "F:\\Documents\\KDL\\MLP_CPP\\data\\01mnist_ytest.txt");
    std::cout << "Load dataset complete!\n";
    std::cout << "Time taken: " << (getTickcount() - time) / 1000. << "\n";
    //// save to diff format
    X_train.save_data("F:\\Documents\\KDL\\MLP_CPP\\data\\01mnist_xtrain.bin");
    X_test.save_data("F:\\Documents\\KDL\\MLP_CPP\\data\\01mnist_xtest.bin");
    y_train.save_data("F:\\Documents\\KDL\\MLP_CPP\\data\\01mnist_ytrain.bin");
    y_test.save_data("F:\\Documents\\KDL\\MLP_CPP\\data\\01mnist_ytest.bin");
    std::cout << "Export complete!\n";

    /*int64_t time = getTickcount();
    std::cout << "Loading dataset, this might take a while...\n";
    Matrix X_train, y_train, X_test, y_test;
    dataset_quick_load(X_train, y_train, X_test, y_test, "F:\\Documents\\KDL\\MLP_CPP\\data\\mnist_xtrain.bin", "F:/Documents/KDL/MLP_CPP\\data\\mnist_ytrain.bin",
        "F:\\Documents\\KDL\\MLP_CPP\\data\\mnist_xtest.bin", "F:\\Documents\\KDL\\MLP_CPP\\data\\mnist_ytest.bin");
    std::cout << "Load dataset complete!\n";
    std::cout << "Time taken: " << (getTickcount() - time) / 1000. << "\n";
    preview_dataset(X_train, y_train, X_test, y_test);*/
}

int main(int argc, char* argv[])
{
    //test_lib();
    //test_mlp();
    //test_grad();
    //test_file();
    //test_svm();
    test_nn();
    return 0;
    // helper
    if (argc < 2)
    {
        std::cout << "Did not receive any parameter! Format for command line input is -> .exe {list of parameters} \n";
        std::cout << "Detailed list of parameters: \n";
        std::cout << "\tmode: {\"0:svm/1:nn\"}, default is 0. You'll need 13 input var for NN and 9 for SVM\nIf use default value, enter random var separate by space\n";
        // neural net configuration
        std::cout << "===================\n";
        std::cout << "Neural net configuration:\n";
        std::cout << "\tModel topology: {\"{input_shape, hidden_1, hidden_2, ..., output_shape}\"}, default is \"{4, 16, 32, 16, 1}\", which is for Iris dataset\n";
        std::cout << "\tLearning rate: {float value}, default is 0.001\n";
        std::cout << "\tEpoch: {int value}, default is 5000\n";
        std::cout << "\tBatch size: {int value}, default is 50\n";
        std::cout << "\tActivation function: {\"activation_func\"}, default is \"relu\", available: [relu, sigmoid]\n";
        std::cout << "\tLoss function: {\"loss_func\"}, default is \"mse\", available: [mse, cross_entropy]\n";
        std::cout << "\tOptimizer: {\"optimizer_name\"}, default is \"sgd\", available: [sgd, adam]\n";
        std::cout << "\tModel's name: {\"model name\"}, default is \"weight\", used to save model's weight, training losses log, prediction result, etc...\n";
        // svm configuration
        std::cout << "===================\n";
        std::cout << "Support vector machine (SVM) configuration:\n";
        std::cout << "\tConstant (margin-sacrifice coef) value: {float value}, default is 1.0\n";
        std::cout << "\tLearning rate: {float value}, default is 0.001\n";
        std::cout << "\tEpoch: {int value}, default is 50\n";
        std::cout << "\tModel's name: {\"model name\"}, default is \"weight\", used to save model's weight, training losses log, prediction result, etc...\n";
        std::cout << "===================\n";
        // data configuration
        std::cout << "Training data configuration:\n";
        std::cout << "\tX_train's filepath: {\"path\"} could be absolute path or relative path\n";
        std::cout << "\ty_train's filepath: {\"path\"} could be absolute path or relative path\n";
        std::cout << "\tX_test's filepath: {\"path\"} could be absolute path or relative path\n";
        std::cout << "\ty_test's filepath: {\"path\"} could be absolute path or relative path\n";
        std::cout << "\tExample: \"../data/mnist_01/X_train.txt\"\n";
        std::cout << "===================\n";
        // example input
        std::cout << "Example: MLP_CPP.exe 0 \"{4, 15, 1}\" 0.001 500 25 \"relu\" \"mse\" \"sgd\" \"iris_default\" \"..\\data\\iris_xtrain.txt\" \"..\\data\\iris_ytrain.txt\" \"..\\data\\iris_xtest.txt\" \"..\\data\\iris_ytest.txt\"\n";
        // note
        std::cout << "Note: if you don't use the default parameters, provide full configuration, I'm too lazy to implement an argument parser. xD\n";
        return 0;
    }
    int runtime_mode = 0;
    double learning_rate = .001f;
    std::string model_name = "weight";
    std::string xtrain_path = "";
    std::string ytrain_path = "";
    std::string xtest_path = "";
    std::string ytest_path = "";

    // nn
    std::string topo = "{4, 16, 32, 16, 2}";
    int epoch_num = 5000;
    int batch_sz = 50;
    std::string activation = "relu";
    std::string loss = "mse";
    std::string optimizer = "sgd";
    // svm
    float C = 1.0;

    if (argc == 5)
    {
        std::cout << "Default configuration, but no\n";
        return 0;
    }
    // program's integrity abomination, i know
    runtime_mode = std::stoi(argv[1]);
    // nn
    if (argc == 14)
    {
        std::cout << "nn\n";
        topo = std::string(argv[2]);
        learning_rate = std::stod(argv[3]);
        epoch_num = std::stoi(argv[4]);
        batch_sz = std::stoi(argv[5]);
        activation = std::string(argv[6]);
        loss = std::string(argv[7]);
        optimizer = std::string(argv[8]);
        model_name = std::string(argv[9]);

        xtrain_path = std::string(argv[10]);        
        ytrain_path = std::string(argv[11]);
        xtest_path = std::string(argv[12]);        
        ytest_path = std::string(argv[13]);

        std::cout << "qq\n";

        // initialize neural net
        NeuralNet net(topo_str_to_vector(topo), learning_rate, epoch_num, batch_sz, activation, loss, optimizer, model_name);

        std::cout << "Loading dataset, this might take a while...\n";
        Matrix X_train, y_train, X_test, y_test;
        dataset_slow_load(X_train, y_train, X_test, y_test, xtrain_path, ytrain_path, xtest_path, ytest_path);
        std::cout << "Load dataset complete!\n";

        net.fit(X_train, y_train);

        net.print_eval(X_test, y_test);

        return 0;
    }
    // svm
    if (argc == 10)
    {
        C = std::stof(argv[2]);
        learning_rate = std::stod(argv[3]);
        epoch_num = std::stoi(argv[4]);
        model_name = std::string(argv[5]);

        xtrain_path = std::string(argv[6]);        
        ytrain_path = std::string(argv[7]);
        xtest_path = std::string(argv[8]);        
        ytest_path = std::string(argv[9]);

        SoftMarginSVM svm(C, learning_rate, epoch_num, model_name);

        std::cout << "Loading dataset, this might take a while...\n";
        Matrix X_train, y_train, X_test, y_test;
        dataset_slow_load(X_train, y_train, X_test, y_test, xtrain_path, ytrain_path, xtest_path, ytest_path);
        std::cout << "Load dataset complete!\n";

        svm.fit(X_train, y_train);

        svm.print_eval(X_test, y_test);

        return 0;
    }

	return 0;
}