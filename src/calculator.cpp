#include <pybind11/pybind11.h>

namespace py = pybind11;

double sum(double a, double b){
    return a+b;
}

double product(double a, double b){
    return a*b;
}

PYBIND11_MODULE(calculator, m){
    m.doc() = "Fast CPP Math Module!!!";
    m.def("sum", &sum, "Adds two numbers");
    m.def("product", &product, "Multiplies two numbers");
}