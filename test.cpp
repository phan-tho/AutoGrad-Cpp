#include "phantho/Tensor.cpp"
#include "phantho/helper.cpp"

using namespace std;

int main() {
    /*
        Just for test
        Note: There is convention that when U wanna calculate `Variable` with
            constant, U must declare constant as a `Variable` with `require_grad=false`
            This limitation in my implement expense 2 times memory than primitive type number.

        Future work:
            + Extend to handle `Tensor`, basic linear algebra operator
            + Traverse all `Variable` in the computation tree (all opimizer need)
            + MLP, RNN, Transformer
            + Convolution 1D, 2D
    */

    // ---------- Test univariable ----------
    // Variable* v1 = new Variable(5, true);
    // Variable* v2 = new Variable(10, true);

    // Variable* v3 = Variable::add(v1, v2);       // v3 = v1 + v2 = 15
    // Variable* v4 = Variable::mul(v3, v3);       // v4 = v3*v3 = 225
    // Variable* v5 = Variable::div(v4, v3);       // v5 = v4/v3 = v3 = v1 + v2 = 15

    // v5->backward();

    // cout << v1->value << ' ' << v2->value << ' ' << v3->value << ' ' << v4->value << ' ' << v5->value << "\n";
    // cout << v1->grad << ' ' << v2->grad << ' ' << v3->grad << ' ' << v4->grad << ' ' << v5->grad << "\n";
    // // 5 10 15 225 15
    // // 1 1 1 0.0666667 1


    // delete v1;
    // delete v2;
    // delete v3;
    // delete v4;
    // delete v5;


    // Done mul tensor. Drawback remaining is allocate and reclaim
    // resource in intermediate step of Mul (vector vars)
    // --------- test multivariable ------------
    int rows = 1, cols = 3;

    double** a1 = random(rows, cols);
    double** a2 = random(cols, rows);

    Tensor* t1 = new Tensor(a1, rows, cols);
    Tensor* t2 = new Tensor(a2, cols, rows);

    Tensor* t3 = Tensor::mul(t1, t2);

    cout << "\nt1 ";
    t1->print();

    cout << "\nt2 ";
    t2->print();

    cout << "\nt3 ";
    t3->print();

    t3->backward();

    cout << "\nt3 ";
    t3->print("grad");

    cout << "\nt1 ";
    t1->print("grad");

    cout << "\nt2 ";
    t2->print("grad");

    delete a1;
    delete a2;
    delete t1;
    delete t2;
    delete t3;
    return 0;
}