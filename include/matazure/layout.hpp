#pragma once

#include <matazure/config.hpp>
#include <matazure/point.hpp>

namespace matazure {

/**
 * \defgroup Tensor Memory Layout
 * @{
 */

template <int_t _Rank>
class column_major_layout {
   public:
    const static int_t rank = _Rank;

    MATAZURE_GENERAL column_major_layout() : column_major_layout(pointi<rank>{0}){};

    MATAZURE_GENERAL column_major_layout(const pointi<rank>& shape) : shape_(shape) {
        stride_[0] = 1;
        for (int_t i = 1; i < rank; ++i) {
            stride_[i] = shape[i - 1] * stride_[i - 1];
        }
        size_ = shape[rank - 1] * stride_[rank - 1];
    }

    MATAZURE_GENERAL column_major_layout(const column_major_layout& rhs)
        : column_major_layout(rhs.shape()) {}

    MATAZURE_GENERAL column_major_layout& operator=(const column_major_layout& rhs) {
        shape_ = rhs.shape();
        stride_ = rhs.stride();
        return *this;
    }

    MATAZURE_GENERAL int_t index2offset(const pointi<rank>& id) const {
        int_t offset = 0;
        for (int_t i = 0; i < rank; ++i) {
            offset += id[i] * stride_[i];
        }

        return offset;
    };

    MATAZURE_GENERAL pointi<rank> offset2index(int_t offset) const {
        pointi<rank> id;
        for (int_t i = rank - 1; i >= 0; --i) {
            id[i] = offset / stride_[i];
            offset = offset % stride_[i];
        }

        return id;
    }

    MATAZURE_GENERAL pointi<rank> shape() const { return shape_; }

    MATAZURE_GENERAL pointi<rank> stride() const { return stride_; }

    MATAZURE_GENERAL int_t size() const { return size_; };

    MATAZURE_GENERAL ~column_major_layout() {}

   private:
    pointi<rank> shape_;
    pointi<rank> stride_;
    int_t size_;
};

// now only suppor layout
template <int_t _Rank>
class row_major_layout {
   public:
    const static int_t rank = _Rank;

    MATAZURE_GENERAL row_major_layout() : row_major_layout(pointi<rank>{0}){};

    MATAZURE_GENERAL row_major_layout(const pointi<rank>& shape) : shape_(shape) {
        stride_[rank - 1] = 1;
        for (int_t i = rank - 2; i >= 0; --i) {
            stride_[i] = shape[i + 1] * stride_[i + 1];
        }
        size_ = stride_[0] * shape[0];
    }

    MATAZURE_GENERAL row_major_layout(const row_major_layout& rhs)
        : row_major_layout(rhs.shape()) {}

    MATAZURE_GENERAL row_major_layout& operator=(const row_major_layout& rhs) {
        shape_ = rhs.shape();
        stride_ = rhs.stride();
        return *this;
    }

    MATAZURE_GENERAL int_t index2offset(const pointi<rank>& id) const {
        typename pointi<rank>::value_type offset = 0;
        for (int_t i = rank - 1; i >= 0; --i) {
            offset += id[i] * stride_[i];
        }
        return offset;
    };

    MATAZURE_GENERAL pointi<rank> offset2index(int_t offset) const {
        pointi<rank> id;
        for (int_t i = 0; i < rank; ++i) {
            id[i] = offset / stride_[i];
            offset = offset % stride_[i];
        }
        return id;
    }

    MATAZURE_GENERAL int_t size() const { return size_; }

    MATAZURE_GENERAL pointi<rank> shape() const { return shape_; }

    MATAZURE_GENERAL pointi<rank> stride() const { return stride_; }

    MATAZURE_GENERAL ~row_major_layout() {}

   private:
    pointi<rank> shape_;
    pointi<rank> stride_;
    int_t size_;
};

struct global_t {};

template <typename _T, int_t _Rank>
struct default_layout {
    typedef row_major_layout<_Rank> type;
};

template <typename _Layout, int_t N>
struct layout_getter;

template <int_t M, int_t N>
struct layout_getter<row_major_layout<M>, N> {
    typedef row_major_layout<N> type;
};

template <int_t M, int_t N>
struct layout_getter<column_major_layout<M>, N> {
    typedef column_major_layout<N> type;
};

namespace internal {
template <int_t _Rank>
inline constexpr pointi<_Rank> get_array_index_by_layout(pointi<_Rank> pt,
                                                         column_major_layout<_Rank>) {
    return reverse_point(pt);
}

template <int_t _Rank>
inline constexpr pointi<_Rank> get_array_index_by_layout(pointi<_Rank> pt,
                                                         row_major_layout<_Rank>) {
    return pt;
}

}  // namespace internal

}  // namespace matazure
