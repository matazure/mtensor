#pragma once

#include <fstream>
#include <iostream>
#include <matazure/tensor.hpp>

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

/**
 * @brief read the binary raw data of a data container(such as tensor, std::vector) from an istream
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
