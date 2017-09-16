#pragma once

namespace matazure { namespace cuda {

namespace puzzle{

template <typename _Func>
inline MATAZURE_GENENRAL void corner_index(pointi<1> origin, pointi<1> extent, _Func func) {
	func(origin);
	func(extent-1);
}

template <typename _Func>
inline MATAZURE_GENENRAL void corner_index(pointi<2> origin, pointi<2> extent, _Func func) {
	func(pointi<2>{origin[0], origin[1]});
	func(pointi<2>{origin[0], extent[1]-1});
	func(pointi<2>{extent[0]-1, origin[1]});
	func(pointi<2>{extent[0]-1, extent[1]-1});
}

template <typename _Func>
inline MATAZURE_GENENRAL void corner_index(pointi<3> origin, pointi<3> extent, _Func func) {
	func(pointi<3>{origin[0], origin[1], origin[2]});
	func(pointi<3>{origin[0], origin[1], extent[2]-1});
	func(pointi<3>{origin[0], extent[1]-1, origin[2]});
	func(pointi<3>{origin[0], extent[1]-1, extent[2]-1});
	func(pointi<3>{extent[0]-1, origin[1], origin[2]});
	func(pointi<3>{extent[0]-1, origin[1], extent[2]-1});
	func(pointi<3>{extent[0]-1, extent[1]-1, origin[2]});
	func(pointi<3>{extent[0]-1, extent[1]-1, extent[2]-1});
}

template <typename _Func>
inline MATAZURE_GENENRAL void corner_index(pointi<4> origin, pointi<4> extent, _Func func) {
	func(pointi<4>{origin[0], origin[1], origin[2], origin[3]});
	func(pointi<4>{origin[0], origin[1], origin[2], extent[3]-1});
	func(pointi<4>{origin[0], origin[1], extent[2]-1, origin[3]});
	func(pointi<4>{origin[0], origin[1], extent[2]-1, extent[3]-1});
	func(pointi<4>{origin[0], extent[1]-1, origin[2], origin[3]});
	func(pointi<4>{origin[0], extent[1]-1, origin[2], extent[3]-1});
	func(pointi<4>{origin[0], extent[1]-1, extent[2]-1, origin[3]});
	func(pointi<4>{origin[0], extent[1]-1, extent[2]-1, extent[3]-1});
	func(pointi<4>{extent[0]-1, origin[1], origin[2], origin[3]});
	func(pointi<4>{extent[0]-1, origin[1], origin[2], extent[3]-1});
	func(pointi<4>{extent[0]-1, origin[1], extent[2]-1, origin[3]});
	func(pointi<4>{extent[0]-1, origin[1], extent[2]-1, extent[3]-1});
	func(pointi<4>{extent[0]-1, extent[1]-1, origin[2], origin[3]});
	func(pointi<4>{extent[0]-1, extent[1]-1, origin[2], extent[3]-1});
	func(pointi<4>{extent[0]-1, extent[1]-1, extent[2]-1, origin[3]});
	func(pointi<4>{extent[0]-1, extent[1]-1, extent[2]-1, extent[3]-1});
}

}

}}
