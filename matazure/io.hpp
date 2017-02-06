#pragma once

#include <fstream>
#include <iostream>
#include <matazure/type_traits.hpp>

namespace matazure {
namespace io {

///TODO：考虑非POD的类型
template <typename _T>
void read_raw_data(std::istream &istr, _T &container) {
	istr.exceptions(std::ifstream::failbit | std::ifstream::badbit | std::ifstream::eofbit);
	istr.read(reinterpret_cast<char *>(container.data()), container.size() * sizeof(typename _T::value_type));
}

template <typename _T>
void write_raw_data(std::ostream &ostr, const _T &container) {
	ostr.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	ostr.write(reinterpret_cast<const char *>(container.data()), container.size() * sizeof(typename _T::value_type));
}

template <typename _T>
void read_raw_data(const std::string &path, _T &container) {
	std::ifstream ifs(path, std::ios::binary);
	read_raw_data(ifs, container);
	//TODO, 判断文件的大小是否合适
	ifs.close();
}

template <typename _T>
void write_raw_data(const std::string &path, const _T &container) {
	std::ofstream ofs(path, std::ios::binary);
	write_raw_data(ofs, container);
	ofs.close();
}

}
}
