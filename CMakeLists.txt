cmake_minimum_required(VERSION 3.28)
project(calculator LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)

find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)
add_subdirectory(third_party/pybind11)
pybind11_add_module(calculator src/calculator.cpp)

execute_process ( COMMAND python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())" OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
install(TARGETS calculator DESTINATION ${PYTHON_SITE_PACKAGES})