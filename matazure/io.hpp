#pragma once

#include <fstream>
#include <iostream>
#include <matazure/type_traits.hpp>

namespace matazure { namespace io{

	///TODO：考虑非POD的类型
	template <typename _T>
	void read(std::istream &is, _T &container) {
		is.read(reinterpret_cast<char *>(container.data()), container.size() * sizeof(typename _T::value_type));
	}

	template <typename _T>
	void write(std::ostream &os, const _T &container) {
		os.write(reinterpret_cast<const char *>(container.data()), container.size() * sizeof(typename _T::value_type));
	}

	template <typename _T>
	void read(const std::string &path, _T &container) {
		std::ifstream ifs(path, std::ios::binary);
		read(ifs, container);
		//TODO, 判断文件的大小是否合适
		ifs.close();
	}

	template <typename _T>
	void write(const std::string &path, const _T &container) {
		std::ofstream ofs(path, std::ios::binary);
		write(ofs, container);
		ofs.close();
	}

}}
