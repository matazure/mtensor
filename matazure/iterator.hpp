#pragma once

#include <iterator>
#include <vector>

namespace matazure {

	template <typename _Tensor>
	class index_iterator {
	public:
		using iterator_category = std::random_access_iterator_tag;
		using value_type = typename _Tensor::value_type;
		using difference_type = int_t;
		using pointer = value_type *;
		using reference = typename _Tensor::reference;

		explicit index_iterator(_Tensor ts, int_t pos) : ts_(ts), pos_(pos) { }
		index_iterator(const index_iterator &other) : ts_(other.ts_), pos_(other.pos_) { }

		index_iterator& operator++() {
			++pos_;
			return *this;
		}

		index_iterator operator++(int) {
			auto retval = *this;
			++(*this);
			return retval;
		}

		index_iterator& operator--() {
			--pos_;
			return *this;
		}

		index_iterator operator--(int) {
			auto retval = *this;
			--(*this);
			return retval;
		}

		///note for performance
		bool operator==(index_iterator other) const { return pos_ == other.pos_; }
		bool operator!=(index_iterator other) const { return pos_ != other.pos_; }
		bool operator>(index_iterator other) const { return pos_ > other.pos_; }
		bool operator<(index_iterator other) const { return pos_ < other.pos_; }
		bool operator>=(index_iterator other) const { return pos_ >= other.pos_; }
		bool operator<=(index_iterator other) const { return pos_ <= other.pos_; }

		const reference operator*() const { return ts_[pos_]; }
		reference operator*() { return ts_[pos_]; }

		index_iterator& operator+=(const difference_type off) {
			pos_ += off;
			return (*this);
		}

		index_iterator operator+(const difference_type off) const {
			index_iterator tmp = *this;
			return (tmp += off);
		}

		index_iterator& operator-=(const difference_type off) {
			return (*this += -off);
		}

		index_iterator operator-(const difference_type off) const {
			index_iterator tmp = *this;
			return (tmp -= off);
		}

		difference_type operator-(const index_iterator& other) const {
			return (pos_ - other.pos_);
		}

		reference operator[](const difference_type off) const {
			return (*(*this + off));
		}

	private:
		_Tensor ts_;
		int_t pos_;
	};

}

namespace matazure {

	template <typename _Tensor>
	auto begin(_Tensor ts) {
		return matazure::index_iterator<_Tensor>(ts, 0);
	}

	template <typename _Tensor>
	auto end( _Tensor ts) {
		return matazure::index_iterator<_Tensor>(ts, ts.size());
	}

	template <typename _Tensor>
	auto rbegin(_Tensor ts) {
		return std::reverse_iterator<decltype(begin(ts))>(end(ts));
	}

	template <typename _Tensor>
	auto rend(_Tensor ts) {
		return std::reverse_iterator<decltype(end(ts))>(begin(ts));
	}

}
