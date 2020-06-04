#pragma once

#include <mtensor.hpp>
#include <type_traits>

using namespace matazure;

namespace matazure {
namespace view {

template <typename _Tensor, int_t _Rank>
struct gradient_op;

template <typename _Tensor>
struct gradient_op<_Tensor, 2> {
    MATAZURE_GENERAL point<typename _Tensor::value_type, _Tensor::rank> operator()(
        point2i idx) const {
        point<typename _Tensor::value_type, 2> re;
        re[0] = image(idx + point2i{1, 0}) - image(idx + point2i{-1, 0});
        re[1] = image(idx + point2i{0, 1}) - image(idx + point2i{0, -1});

        return re;
    }

    _Tensor image;
};

template <typename _Tensor>
struct gradient_op<_Tensor, 3> {
    MATAZURE_GENERAL point<typename _Tensor::value_type, _Tensor::rank> operator()(
        point3i idx) const {
        point<typename _Tensor::value_type, 3> re;
        re[0] = image(idx + point3i{1, 0, 0}) - image(idx + point3i{-1, 0, 0});
        re[1] = image(idx + point3i{0, 1, 0}) - image(idx + point3i{0, -1, 0});
        re[2] = image(idx + point3i{0, 0, 1}) - image(idx + point3i{0, 0, -1});
        return re;
    }

    _Tensor image;
};

template <typename image_type>
inline auto gradient(image_type image)
    -> decltype(make_lambda(image.shape(), gradient_op<image_type, image_type::rank>{image},
                            runtime_t<image_type>{})) {
    using value_type = typename image_type::value_type;
    // static const int_t rank = image_type::rank;

    return make_lambda(image.shape(), gradient_op<image_type, image_type::rank>{image},
                       runtime_t<image_type>{});
}

template <typename _Tensor, int_t _Rank>
struct div_op;

template <typename _Tensor>
struct div_op<_Tensor, 2> {
    MATAZURE_GENERAL typename _Tensor::value_type::value_type operator()(point2i idx) const {
        auto nxx = image(idx + point2i{1, 0})[0] - image(idx + point2i{-1, 0})[0];
        auto nyy = image(idx + point2i{0, 1})[1] - image(idx + point2i{0, -1})[1];
        return nxx + nyy;
    }

    _Tensor image;
};

template <typename _Tensor>
struct div_op<_Tensor, 3> {
    MATAZURE_GENERAL typename _Tensor::value_type::value_type operator()(point3i idx) const {
        auto nxx = image(idx + point3i{1, 0, 0})[0] - image(idx + point3i{-1, 0, 0})[0];
        auto nyy = image(idx + point3i{0, 1, 0})[1] - image(idx + point3i{0, -1, 0})[1];
        auto nzz = image(idx + point3i{0, 0, 1})[2] - image(idx + point3i{0, 0, -1})[2];
        return nxx + nyy + nzz;
    }

    _Tensor image;
};

template <typename image_type>
inline auto div(image_type image)
    -> decltype(make_lambda(image.shape(), div_op<image_type, image_type::rank>{image},
                            runtime_t<image_type>{})) {
    // static const int_t rank = image_type::rank;
    typedef typename image_type::value_type image_value_type;
    typedef typename image_value_type::value_type value_type;

    return make_lambda(image.shape(), div_op<image_type, image_type::rank>{image},
                       runtime_t<image_type>{});
}

template <typename _Tensor, int_t _Rank>
struct laplace_op;

template <typename _Tensor>
struct laplace_op<_Tensor, 2> {
    MATAZURE_GENERAL typename _Tensor::value_type operator()(point2i idx) const {
        auto tmp = -4 * image(idx);
        tmp = tmp + image(idx + pointi<2>{-1, 0});
        tmp = tmp + image(idx + pointi<2>{1, 0});
        tmp = tmp + image(idx + pointi<2>{0, -1});
        tmp = tmp + image(idx + pointi<2>{0, 1});
        return tmp;
    }

    _Tensor image;
};

template <typename _Tensor>
struct laplace_op<_Tensor, 3> {
    MATAZURE_GENERAL typename _Tensor::value_type operator()(point3i idx) const {
        auto tmp = -6 * image(idx);
        tmp = tmp + image(idx + pointi<3>{-1, 0, 0});
        tmp = tmp + image(idx + pointi<3>{1, 0, 0});
        tmp = tmp + image(idx + pointi<3>{0, -1, 0});
        tmp = tmp + image(idx + pointi<3>{0, 1, 0});
        tmp = tmp + image(idx + pointi<3>{0, 0, -1});
        tmp = tmp + image(idx + pointi<3>{0, 0, 1});
        return tmp;
    }

    _Tensor image;
};

template <typename image_type>
auto laplace(image_type image)
    -> decltype(make_lambda(image.shape(), laplace_op<image_type, image_type::rank>{image},
                            runtime_t<image_type>{})) {
    using value_type = typename image_type::value_type;
    // static const int_t rank = image_type::rank;
    return make_lambda(image.shape(), laplace_op<image_type, image_type::rank>{image},
                       runtime_t<image_type>{});
}

}  // namespace view
}  // namespace matazure

template <typename _ValueType, int_t _Rank>
MATAZURE_GENERAL inline point<_ValueType, _Rank> normalize(const point<_ValueType, _Rank>& vec) {
    _ValueType sum_norm2 = 0;
    for (int_t i = 0; i < _Rank; ++i) {
        sum_norm2 += vec[i] * vec[i];
    }

    _ValueType scale = sqrt(sum_norm2);

    point<_ValueType, _Rank> re;
    for (int_t i = 0; i < _Rank; ++i) {
        re[i] = vec[i] / (scale + 1e-10);
    }

    return re;
}

template <typename image_type>
inline image_type make_border_copy(image_type img, std::integral_constant<int_t, 2>) {
    //
    static const int_t rank = image_type::rank;
    pointi<rank> padding;
    fill(padding, 1);

    image_type img_border(img.shape() + padding * 2);
    copy(img, view::slice(img_border, padding, img.shape()));

    auto shape = img_border.shape();
    copy(view::gather<0>(img_border, 1), view::gather<0>(img_border, 0));
    copy(view::gather<0>(img_border, shape[0] - 2), view::gather<0>(img_border, shape[0] - 1));
    copy(view::gather<1>(img_border, 1), view::gather<1>(img_border, 0));
    copy(view::gather<1>(img_border, shape[1] - 2), view::gather<1>(img_border, shape[1] - 1));

    return img_border;
}

template <typename image_type>
inline image_type make_border_copy(image_type img, std::integral_constant<int_t, 3>) {
    //
    static const int_t rank = image_type::rank;
    pointi<rank> padding;
    fill(padding, 1);

    image_type img_border(img.shape() + padding * 2);
    copy(img, view::slice(img_border, padding, img.shape()));

    auto shape = img_border.shape();

    copy(view::gather<0>(img_border, 1), view::gather<0>(img_border, 0));
    copy(view::gather<0>(img_border, shape[0] - 2), view::gather<0>(img_border, shape[0] - 1));
    copy(view::gather<1>(img_border, 1), view::gather<1>(img_border, 0));
    copy(view::gather<1>(img_border, shape[1] - 2), view::gather<1>(img_border, shape[1] - 1));
    copy(view::gather<2>(img_border, 1), view::gather<2>(img_border, 0));
    copy(view::gather<2>(img_border, shape[2] - 2), view::gather<2>(img_border, shape[2] - 1));

    return img_border;
}

template <typename image_type>
inline auto gradient(image_type image) -> decltype(
    view::gradient(view::slice(image, pointi<image_type::rank>{}, image.shape())).persist()) {
    auto image_with_border =
        make_border_copy(image, std::integral_constant<int_t, image_type::rank>{});
    pointi<image_type::rank> padding;
    fill(padding, 1);
    return view::gradient(view::slice(image_with_border, padding, image.shape())).persist();
}

template <typename image_type>
inline auto div(image_type image) -> decltype(
    view::div(view::slice(image, pointi<image_type::rank>{}, image.shape())).persist()) {
    auto image_with_border =
        make_border_copy(image, std::integral_constant<int_t, image_type::rank>{});
    pointi<image_type::rank> padding;
    fill(padding, 1);
    return view::div(view::slice(image_with_border, padding, image.shape())).persist();
}

template <typename image_type>
inline image_type laplace(image_type image) {
    auto image_with_border =
        make_border_copy(image, std::integral_constant<int_t, image_type::rank>{});
    pointi<image_type::rank> padding;
    fill(padding, 1);
    return view::laplace(view::slice(image_with_border, padding, image.shape())).persist();
}

template <typename image_type>
inline void neumann_bound_conf(image_type image, std::integral_constant<int_t, 2>) {
    auto end = image.shape() - 1;
    // border
    copy(view::gather<0>(image, 2), view::gather<0>(image, 0));
    copy(view::gather<0>(image, end[0] - 2), view::gather<0>(image, end[0]));
    copy(view::gather<1>(image, 2), view::gather<1>(image, 0));
    copy(view::gather<1>(image, end[1] - 2), view::gather<1>(image, end[1]));

    // corner bug with cuda, not effect result
    // image(0,      0)        = image(2,          2);
    // image(end[0], 0)        = image(end[0] - 2, 2);
    // image(0,      end[1])   = image(2,          end[1] - 2);
    // image(end[0], end[1])   = image(end[0] - 2, end[1] - 2);
}

template <typename image_type>
inline void neumann_bound_conf(image_type image, std::integral_constant<int_t, 3>) {
    auto end = image.shape() - 1;
    // border
    copy(view::gather<0>(image, 2), view::gather<0>(image, 0));
    copy(view::gather<0>(image, end[0] - 2), view::gather<0>(image, end[0]));
    copy(view::gather<1>(image, 2), view::gather<1>(image, 0));
    copy(view::gather<1>(image, end[1] - 2), view::gather<1>(image, end[1]));
    copy(view::gather<2>(image, 2), view::gather<2>(image, 0));
    copy(view::gather<2>(image, end[2] - 2), view::gather<2>(image, end[2]));
}

template <typename image_type>
image_type drlse_edge(image_type mat_phi0, image_type mat_g, float lambda, float mu, float alpha,
                      float epsilon, float timestep, int counter = 10) {
    typedef typename image_type::value_type value_type;
    static const size_t rank = image_type::rank;

    for_each(mat_g.shape() == mat_phi0.shape(),
             [](bool e) { MATAZURE_ASSERT(e, "shape must be matched"); });

    auto mat_g_grad = gradient(mat_g);
    auto mat_phi = identify(mat_phi0, runtime_t<image_type>{});

    while (counter--) {
        neumann_bound_conf(mat_phi, std::integral_constant<int_t, rank>{});
        auto mat_phi_grad = gradient(mat_phi);
        auto mat_normalized_gradient_phi =
            view::map(mat_phi_grad,
                      [] MATAZURE_GENERAL(point<value_type, rank> p) { return normalize(p); })
                .persist();
        auto mat_curvature = div(mat_normalized_gradient_phi);
        auto mat_dist_term = laplace(mat_phi) - mat_curvature;

        auto mat_dirac_phi = view::map(mat_phi, [epsilon] MATAZURE_GENERAL(value_type x) {
            if (std::abs(x) > epsilon) return 0.0f;
            return (1.0f / (2 * epsilon)) * (1.0f + std::cos(3.1415926f * x / epsilon));
        });
        auto mat_area_term = mat_dirac_phi * mat_g;

        auto mat_temp = mat_g_grad * mat_phi_grad;
        auto mat_temp_sum = view::map(mat_temp, [] MATAZURE_GENERAL(point<value_type, rank> x) {
            auto tmp = zero<value_type>::value();
            for (int_t i = 0; i < x.size(); ++i) {
                tmp = tmp + x[i];
            }
            return tmp;
        });
        auto mat_edge_term = mat_dirac_phi * mat_temp_sum + mat_area_term * mat_curvature;

        auto mat_phi_next = (mat_phi + timestep * (mu * mat_dist_term + lambda * mat_edge_term +
                                                   alpha * mat_area_term))
                                .persist();
        copy(mat_phi_next, mat_phi);
    }

    return mat_phi;
}

/*

function phi = drlse_edge(phi_0, g, lambda,mu, alfa, epsilon, timestep, iter, potentialFunction)
%  This Matlab code implements an edge-based active contour model as an
%  application of the Distance Regularized Level Set Evolution (DRLSE) formulation in Li et al's
paper:
%
%      C. Li, C. Xu, C. Gui, M. D. Fox, "Distance Regularized Level Set Evolution and Its
Application to Image Segmentation", %        IEEE Trans. Image Processing, vol. 19 (12),
pp.3243-3254, 2010.
%
%  Input:
%      phi_0: level set function to be updated by level set evolution
%      g: edge indicator function
%      mu: weight of distance regularization term
%      timestep: time step
%      lambda: weight of the weighted length term
%      alfa:   weight of the weighted area term
%      epsilon: width of Dirac Delta function
%      iter: number of iterations
%      potentialFunction: choice of potential function in distance regularization term.
%              As mentioned in the above paper, two choices are provided:
potentialFunction='single-well' or %              potentialFunction='double-well', which correspond
to the potential functions p1 (single-well) %              and p2 (double-well), respectively.% %
Output: %      phi: updated level set function after level set evolution
%
% Author: Chunming Li, all rights reserved
% E-mail: lchunming@gmail.com
%         li_chunming@hotmail.com
% URL:  http://www.engr.uconn.edu/~cmli/

phi=phi_0;
[vx, vy]=gradient(g);
for k=1:iter
    %phi=NeumannBoundCond(phi);
    [phi_x,phi_y]=gradient(phi);
    s=sqrt(phi_x.^2 + phi_y.^2);
    smallNumber=1e-10;
    Nx=phi_x./(s+smallNumber); % add a small positive number to avoid division by zero
    Ny=phi_y./(s+smallNumber);
    curvature=div(Nx,Ny);
    if strcmp(potentialFunction,'single-well')
        distRegTerm = 4*del2(phi)-curvature;  % compute distance regularization term in equation
(13) with the single-well potential p1. elseif strcmp(potentialFunction,'double-well');
        distRegTerm=distReg_p2(phi);  % compute the distance regularization term in eqaution (13)
with the double-well potential p2. else disp('Error: Wrong choice of potential function. Please
input the string "single-well" or "double-well" in the drlse_edge function.'); end
    diracPhi=Dirac(phi,epsilon);
    areaTerm=diracPhi.*g; % balloon/pressure force
    edgeTerm=diracPhi.*(vx.*Nx+vy.*Ny) + diracPhi.*g.*curvature;
    phi=phi + timestep*(mu*distRegTerm + lambda*edgeTerm + alfa*areaTerm);
end

end


function f = distReg_p2(phi)
% compute the distance regularization term with the double-well potential p2 in eqaution (16)
[phi_x,phi_y]=gradient(phi);
s=sqrt(phi_x.^2 + phi_y.^2);
a=(s>=0) & (s<=1);
b=(s>1);
ps=a.*sin(2*pi*s)/(2*pi)+b.*(s-1);  % compute first order derivative of the double-well potential p2
in eqaution (16) dps=((ps~=0).*ps+(ps==0))./((s~=0).*s+(s==0));  % compute d_p(s)=p'(s)/s in
equation (10). As s-->0, we have d_p(s)-->1 according to equation (18) f = div(dps.*phi_x - phi_x,
dps.*phi_y - phi_y) + 4*del2(phi); end

function f = div(nx,ny)
[nxx,junk]=gradient(nx);
[junk,nyy]=gradient(ny);
f=nxx+nyy;
end

function f = Dirac(x, sigma)
f=(1/2/sigma)*(1+cos(pi*x/sigma));
b = (x<=sigma) & (x>=-sigma);
f = f.*b;
end

function g = NeumannBoundCond(f)
% Make a function satisfy Neumann boundary condition
[nrow,ncol] = size(f);
g = f;
g([1 nrow],[1 ncol]) = g([3 nrow-2],[3 ncol-2]);
g([1 nrow],2:end-1) = g([3 nrow-2],2:end-1);
g(2:end-1,[1 ncol]) = g(2:end-1,[3 ncol-2]);
end

*/
