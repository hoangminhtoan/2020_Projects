#include <dlib/matrix.h>
#include <iostream>

int main(){
    // definitions
    {
        // compile time sized matrix
        dlib::matrix<double, 3, 1> y;
        // dynamically size matrix
        dlib::matrix<double> m(3, 3);
        // later we can change size of this matrix
        m.set_size(6, 6);
    }
    // initializations
    {
        // comma operator
        dlib::matrix<double> m(3, 3);
        m = 1., 2., 3., 4., 5., 6., 7., 8., 9.;
        std::cout << "Matrix from comma operator\n" << m << std::endl;

        // wrap array
        double data[] = {1, 2, 3, 4, 5, 6};
        auto m2 = dlib::mat(data, 2, 3); // create matrix with size 2x3
        std::cout << "Matrix from array\n" << m2 << std::endl;

        // Matrix elements can be accessed with () operator
        m(1, 2) = 300;
        std::cout << "Matrix element updated\n" << m << std::endl;

        // Also you can initialize matrix with some predefined values
        auto a = dlib::identity_matrix<double>(3);
        std::cout << "Identity matrix\n" << a << std::endl;

        auto b = dlib::ones_matrix<double>(3, 4);
        std::cout << "Ones matrix\n" << b << std::endl;

        auto c = dlib::randm(3, 4); // Matrix with random values with size 3x4
        std::cout << "Random matrix\n" << c << std::endl;

    }
    return 0;
}