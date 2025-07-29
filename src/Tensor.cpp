#include "Variable.cpp"

class Tensor {
public:
    Variable*** tensor;
    bool require_grad;
    int n_rows, n_columns;

    Tensor(bool require_grad=true) {
        this->require_grad = require_grad;
    }

    Tensor(int n_rows, int n_columns, bool require_grad=true) {
        this->require_grad = require_grad;
        this->n_rows = n_rows;
        this->n_columns = n_columns;

        tensor = new Variable**[n_rows];
        for (int i = 0; i < n_rows; i++) {
            *(tensor + i) = new Variable*[n_columns];
        }
    }

    Tensor(double** array, int n_rows, int n_columns, bool require_grad=true) {
        this->require_grad = require_grad;
        this->n_rows = n_rows;
        this->n_columns = n_columns;

        tensor = new Variable**[n_rows];
        for (int i = 0; i < n_rows; i++) {
            *(tensor + i) = new Variable*[n_columns];
            for (int j = 0; j < n_columns; j++) {
                tensor[i][j] = new Variable(array[i][j], require_grad);
            }
        }
    }

    ~Tensor() {
        tensor = nullptr;
    }

    void print(const std::string& obj="value") {
        std::cout << obj << "\n";
        for (int i = 0; i < n_rows; i++) {
            for (int j = 0; j < n_columns; j++) {
                std::cout << ((obj == "value") ? tensor[i][j]->value : tensor[i][j]->grad) << ' ';
            }
            std::cout << "\n";
        }
    }

    void backward(double grad_par=1) {
        if (!this->require_grad)        return;
        for (int i = 0; i < n_rows; i++) {
            for (int j = 0; j < n_columns; j++) {
                tensor[i][j]->backward(grad_par);
            }
        }
    }

    static Tensor* add(Tensor* v1, Tensor* v2);
    static Tensor* mul(Tensor* v1, Tensor* v2);
};

Tensor* Tensor::add(Tensor* t1, Tensor* t2) {
    assert((t1->n_rows == t2->n_rows && t1->n_columns == t2->n_columns), 
    "2 tensor must same shape but found t1: ()");

    int n_rows = t1->n_rows;
    int n_cols = t1->n_columns;
    Tensor* newTensor = new Tensor(n_rows, n_cols, t1->require_grad || t2->require_grad);

    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            Variable* tmp = newTensor->tensor[i][j];
            newTensor->tensor[i][j] = Variable::add(t1->tensor[i][j], t2->tensor[i][j]);
            delete tmp;
        }
    }

    return newTensor;
}

Tensor* Tensor::mul(Tensor* t1, Tensor* t2) {
    assert(t1->n_columns == t2->n_rows, "conflict shape");

    int n_rows = t1->n_rows;
    int n_cols = t2->n_columns;
    int n_loops = t1->n_columns;
    Tensor* newTensor = new Tensor(n_rows, n_cols, t1->require_grad || t2->require_grad);

    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            // hang i t1, cot j t2
            std::vector<IgnoreVariable*> vars;      // hmm, how to delete this
            for (int k = 0; k < n_loops; k++) {
                vars.push_back(Variable::mul(t1->tensor[i][k], t2->tensor[k][j]));
            }
            Variable* tmp = newTensor->tensor[i][j];
            newTensor->tensor[i][j] = Variable::add(vars);
            delete tmp;
        }
    }

    return newTensor;
}

// Tensor* Tensor::mul(Tensor* v1, Tensor* v2) {
//     Mul* mul = new Mul(v1, v2);

//     Tensor* newVar = new Tensor(mul->compute());
//     newVar->fn = mul;

//     return newVar;
// }