#include "Operator.cpp"

class Variable : public IgnoreVariable {
public:
    Operator* fn;

    Variable(double value=0, bool require_grad=true) : IgnoreVariable(value, require_grad) {
        fn = NULL;
    }

    ~Variable() {
        fn= nullptr;
    }

    // cast, alike ignorant but important
    Variable(IgnoreVariable* v) {
        this->value = v->value;
        this->grad = v->grad;
        this->require_grad = v->require_grad;
        fn = NULL;    // dcm quen cho nay nhin loi ca mat
    }

    // do operator on 2 variable and push grad_function to res (new variable)
    static Variable* add(Variable* v1, Variable* v2);
    static Variable* add(std::vector<IgnoreVariable*> vars);
    static Variable* mul(Variable* v1, Variable* v2);
    static Variable* div(Variable* v1, Variable* v2);

    // cumulative backward
    void backward(double grad_par=1) {
        if (this->require_grad)     this->grad += grad_par;

        // if (fn != nullptr)   std::cout << "HI" << std::endl;
        if (fn != NULL)             fn->backward(grad_par);
    }
};

Variable* Variable::add(Variable* v1, Variable* v2) {
    Add* add = new Add(v1, v2);

    Variable* newVar = new Variable(add->compute());
    newVar->fn = add;

    return newVar;
}

Variable* Variable::add(std::vector<IgnoreVariable*> vars) {
    Add* add = new Add(vars);

    Variable* newVar = new Variable(add->compute());
    newVar->fn = add;

    return newVar;
}

Variable* Variable::mul(Variable* v1, Variable* v2) {
    Mul* mul = new Mul(v1, v2);

    Variable* newVar = new Variable(mul->compute());
    newVar->fn = mul;

    return newVar;
}

Variable* Variable::div(Variable* v1, Variable* v2) {
    Div* div = new Div(v1, v2);

    Variable* newVar = new Variable(div->compute());
    newVar->fn = div;

    return newVar;
}