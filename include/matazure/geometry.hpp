#pragma once

#include <matazure/algorithm.hpp>

namespace matazure {

/**
 * @brief detect whether a point is inside_rect of a rect (left close, right open)
 * @param idx point position
 * @param origin the lef top index of the rect
 * @param rect the rect
 * @return returns true if the point is inside_rect of the rect
 */
template <int_t _Rank>
inline MATAZURE_GENERAL bool inside_rect(point<int_t, _Rank> idx, point<int_t, _Rank> origin,
                                         point<int_t, _Rank> rect) {
    for (int_t i = 0; i < _Rank; ++i) {
        // if (idx[i] < origin[i] || idx[i] >= origin[i] + rect[i]) return false;
        if (static_cast<uint_t>(idx[i] - origin[i]) >= static_cast<uint_t>(rect[i])) return false;
    }

    return true;
}

template <typename _Fun>
inline void for_border(const pointi<2>& extent, const pointi<2>& origin_padding,
                       const pointi<2>& end_padding, _Fun fun) {
    // top
    for (int j = -origin_padding[1]; j < 0; ++j) {
        for (int i = -origin_padding[0]; i < extent[0] + end_padding[0]; ++i) {
            fun(pointi<2>{i, j});
        }
    }

    // center
    for (int j = 0; j < extent[1]; ++j) {
        for (int i = -origin_padding[0]; i < 0; ++i) {
            fun(pointi<2>{i, j});
        }
        for (int i = extent[0]; i < extent[0] + end_padding[0]; ++i) {
            fun(pointi<2>{i, j});
        }
    }

    // bottome
    for (int j = extent[1]; j < extent[1] + end_padding[1]; ++j) {
        for (int i = -origin_padding[0]; i < extent[0] + end_padding[0]; ++i) {
            fun(pointi<2>{i, j});
        }
    }
}

}  // namespace matazure
