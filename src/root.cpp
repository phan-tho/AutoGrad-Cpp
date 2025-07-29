#include <iostream>
#include <vector>

class IgnoreVariable {
public:
    double value;
    bool require_grad;
    double grad;

    IgnoreVariable(double value=0, bool require_grad=true) {
        grad = 0;
        this->value = value;
        this->require_grad = require_grad;
    }

    // IgnoreVariable operator+(IgnoreVariable other);
    virtual void backward(double grad_par=1) { }
};

void assert(const bool cond, const std::string& text) {
    if (!cond) {
        throw std::runtime_error(text);
    }
}