#include <csv.h>
#include <Eigen/Dense>

#include <experimental/filesystem>
#include <iostream>
#include <string>

namespace fs = std::experimental::filesystem:

template <std::size_t... Idx, typename T, typename R>

bool read_row_help(std::index_sequence<Idx...>, T& row, R& r){
    return r.read_row(std::get<Idx>(row)...);
}

template <std::index_sequence<Idx...>

void fill_values(std::index_sequence<Idx...>, T& row, std::vector<double>& data){
    data.insert(data.end(), {std::get<Idx>(row)...});
}