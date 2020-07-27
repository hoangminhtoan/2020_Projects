#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xeval.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>

#include <iostream>
#include <vector>


int main(){
    {
        // declaration of dynamically sized array
        {
            std::vector<uint64_t> shape = {3, 2, 4};
            xt::xarray<double, xt::layout_type::row_major> a(shape);
        }
        // declaration of dynamically sized tensor with fixed dimmentions number
        {
            std::array<size_t, 3> shape = {3, 2, 4};
            xt::xtensor<double, 3> a(shape);
        }

        // declaration of tensor with shape fixed at compile time
        {
            xt::xtensor_fixed<double, xt::xshape<3, 2, 4>> a;
        }

        // Initialization of xtensor arrays can be done with C++ initializer lists
        {
            xt::xarray<double> arr1{{1.0, 2.0, 3.0}, {2.0, 4.0, 6.0}, {3.0, 5.0, 7.0}};
            std::cout << "Tensor from initializer list :\n" << arr1 << std::endl;
        }

        // Special types of initlaizers
        {
            std::vector<uint64_t> shape = {2, 2};
            std::cout << "Ones matrix :\n" << xt::ones<float>(shape) << std::endl;
            std::cout << "Zeros matrix: \n" << xt::zeros<float>(shape) << std::endl;
            std::cout << "Matrix with ones on the diagonal: \n" << xt::eye<float>(shape) << std::endl;
        }
        // element access
        {
            std::vector<size_t> shape = {3, 2, 4};
            xt::xarray<float> a = xt::ones<float>(shape);
            a(2, 1, 3) = 3.14f;
            std::cout << "Updated element :\n" << a << std::endl;
        }
        // arithmetic operations examples
        {
            xt::xarray<double> a = xt::random::rand<double>({2, 2});
            xt::xarray<double> b = xt::random::rand<double>({2, 2});

            std::cout << "A: \n" << a << std::endl;
            std::cout << "B: \n" << b << std::endl;
        }
    }
    return 0;
}