#pragma once

#include <fstream>
#include <iostream>
#include <matazure/algorithm.hpp>
#include <matazure/tensor.hpp>
#include <matazure/view/view.hpp>

namespace matazure {

template <class _ValueType, int_t _Rank>
inline std::ostream& operator<<(std::ostream& out, const point<_ValueType, _Rank>& p) {
    out << "{";
    for (int_t i = 0; i < p.size(); ++i) {
        out << p[i];
        if (i != p.size() - 1) {
            out << ", ";
        }
    }
    out << "}";

    return out;
}

template <typename _Tensor, int_t _Rank>
struct printer {
    static void print(std::ostream& out, _Tensor ts) {
        out << "{";
        for (int_t i = 0; i < ts.shape(0); ++i) {
            auto tmp_view = view::gather<0>(ts, i);
            printer<decltype(tmp_view), _Rank - 1>::print(out, tmp_view);
            if (i != ts.shape(0) - 1) {
                out << ", " << std::endl;
            }
        }
        out << "}";
    }

};  // namespace matazure

template <typename _Tensor>
struct printer<_Tensor, 1> {
    static void print(std::ostream& out, _Tensor ts) {
        out << "{";
        for (int_t i = 0; i < ts.size(); ++i) {
            out << ts(i);
            if (i != ts.size() - 1) {
                out << ", ";
            }
        }
        out << "}";
    }
};

//目前仅支持
template <typename _Tensor>
inline std::ostream& operator<<(std::ostream& out, const tensor_expression<_Tensor>& e_ts) {
    // auto p = e_ts();
    auto ts = e_ts();
    printer<_Tensor, _Tensor::rank>::print(out, ts);
    return out;
}

/**
 * @brief read the binary raw data of a data container(such as tensor, std::vector) from an
 istream
 * @param istr an input istream
 * @param container dest container
 * @todo 考虑非POD的类型
 */
template <typename _T>
void read_raw_data(std::istream& istr, _T& container) {
    istr.exceptions(std::ifstream::failbit | std::ifstream::badbit | std::ifstream::eofbit);
    istr.read(reinterpret_cast<char*>(container.data()),
              container.size() * sizeof(decltype((*container.data()))));
}

/**
 * @brief write binary raw data of a data container(such as tensor, std::vector) to an istream
 * @param ostr an output istream
 * @param container source container
 */
template <typename _T>
void write_raw_data(std::ostream& ostr, const _T& container) {
    ostr.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    ostr.write(reinterpret_cast<const char*>(container.data()),
               container.size() * sizeof(decltype((*container.data()))));
}

// /**
//  * @brief read the binary raw data of a data container(such as tensor, std::vector) from an
//  istream
//  * @param istr an input istream
//  * @param container dest container
//  * @todo 考虑非POD的类型
//  */
// template <typename _T>
// void read_raw_data(std::istream& istr, _T& container) {
//     istr.exceptions(std::ifstream::failbit | std::ifstream::badbit | std::ifstream::eofbit);
//     istr.read(reinterpret_cast<char*>(container.data()),
//               container.size() * sizeof(decltype((*container.data()))));

//     for_each(container, [&](typename _T::value_type& v) {
//         istr.read(reinterpret_cast<char*>(&v), sizeof(v));
//     });
// }

// /**
//  * @brief write binary raw data of a data container(such as tensor, std::vector) to an istream
//  * @param ostr an output istream
//  * @param container source container
//  */
// template <typename _T>
// void write_raw_data(std::ostream& ostr, const _T& container) {
//     ostr.exceptions(std::ifstream::failbit | std::ifstream::badbit);
//     for_each(container, [&](typename _T::value_type& v) {
//         ostr.write(reinterpret_cast<char*>(&v), sizeof(v));
//     });
// }

/**
 * @brief read the binary raw data of a data container(such as tensor, std::vector) from a file
 * @param path source file
 * @param container dest container
 * @todo 考虑非POD的类型
 */
template <typename _T>
void read_raw_data(const std::string& path, _T& container) {
    std::ifstream ifs(path, std::ios::binary);
    read_raw_data(ifs, container);
    // TODO, 判断文件的大小是否合适
    ifs.close();
}

/**
 * @brief write binary raw data of a data container(such as tensor, std::vector) to a file
 * @param path dest file
 * @param container source container
 */
template <typename _T>
void write_raw_data(const std::string& path, const _T& container) {
    std::ofstream ofs(path, std::ios::binary);
    write_raw_data(ofs, container);
    ofs.close();
}

}  // namespace matazure
