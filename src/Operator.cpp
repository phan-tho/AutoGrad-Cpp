#include "root.cpp"

class Operator {
protected:
    // IgnoreVariable **vars;
    // Dung con tro 2 chieu loi ra cai bug to dung
    // Mang la ctdl tuyen tinh ==> cast don 1 phan tu thi duoc
    // nhwng deo cast ca mang duoc

    std::vector<IgnoreVariable*> vars;
public:

    Operator(IgnoreVariable* v1, IgnoreVariable* v2) { //: v1(v1), v2(v2) {}
        vars.push_back(v1);
        vars.push_back(v2);
        // vars = new IgnoreVariable*[2];

        // IgnoreVariable* tmp1 = vars[0], *tmp2 = vars[1];

        // vars[0] = v1;
        // vars[1] = v2;

        // delete tmp1;
        // delete tmp2;
    }

    Operator(std::vector<IgnoreVariable*> vars) : vars(vars) {}

    ~Operator() {
        for (auto p : vars) {
            p = nullptr;
        }
    }

    virtual IgnoreVariable* compute() { return vars[0]; } 

    virtual void backward(double grad_par) {    }
};

class Add : public Operator {
public:
    Add(IgnoreVariable* v1, IgnoreVariable* v2) : Operator(v1, v2) {}
    Add(std::vector<IgnoreVariable*> vars) : Operator(vars) {}

    IgnoreVariable* compute() {
        bool grad_required = 0; //v1->require_grad || v2->require_grad;
        double val = 0;

        for (int i = 0; i < vars.size(); i++) {
            grad_required |= vars[i]->require_grad;
            val += vars[i]->value;
        }
        IgnoreVariable* newv = new IgnoreVariable(val, grad_required);
        return newv;
    }

    void backward(double grad_par) {
        for (int i = 0; i < vars.size(); i++) {
            vars[i]->backward(grad_par);
        }

        // v1->backward(grad_par);
        // v2->backward(grad_par);
    }
};

class Mul : public Operator {
public:
    Mul(IgnoreVariable* v1, IgnoreVariable* v2) : Operator(v1, v2) {}
    Mul(std::vector<IgnoreVariable*> vars) : Operator(vars) {}

    IgnoreVariable* compute() {
        // bool grad_required = v1->require_grad || v2->require_grad;
        // IgnoreVariable* newv = new IgnoreVariable(v1->value * v2->value, grad_required);

        bool grad_required = 0; //v1->require_grad || v2->require_grad;
        double val = 1;

        for (int i = 0; i < vars.size(); i++) {
            grad_required |= vars[i]->require_grad;
            val *= vars[i]->value;
        }
        IgnoreVariable* newv = new IgnoreVariable(val, grad_required);
        return newv;
    }

    void backward(double grad_par) {            // grad_par = 1; v4 = 30, v3 = 15
        // v1->backward(grad_par * v2->value);
        // v2->backward(grad_par * v1->value);

        for (int i = 0; i < vars.size(); i++) {
            double c_grad = grad_par;

            for (int j = 0; j < vars.size(); j++) {
                c_grad *= (i == j ? 1 : vars[j]->value);
            } 

            vars[i]->backward(c_grad);
        }
    }
};

class Div : public Operator {
public:
    Div(IgnoreVariable* v1, IgnoreVariable* v2) : Operator(v1, v2) {}
    Div(std::vector<IgnoreVariable*> vars) : Operator(vars) {}

    IgnoreVariable* compute() {
        bool grad_required = vars[0]->require_grad || vars[1]->require_grad;
        if (vars[1]->value == 0) {
            vars[1]->value += 1e-6;
        }
        IgnoreVariable* newv = new IgnoreVariable(vars[0]->value / vars[1]->value, grad_required);
        return newv;
    }

    void backward(double grad_par) {           // vars[0] / vars[1]
        vars[0]->backward(grad_par / vars[1]->value);
        vars[1]->backward(grad_par * (-vars[0]->value / (vars[1]->value * vars[1]->value)));
    }
};