#pragma once

#include <mtensor.hpp>
#include "image_helper.hpp"

namespace matazure {
namespace view {

template <typename image_type>
inline auto gradient(image_type image) {
    using value_type = typename image_type::value_type;
    static const int_t rank = image_type::rank;

    return make_lambda(image.shape(), [=](pointi<rank> idx) {
        point<value_type, rank> re;
        re[0] = image(idx + pointi<rank>{1, 0}) - image(idx + pointi<rank>{-1, 0});
        re[1] = image(idx + pointi<rank>{0, 1}) - image(idx + pointi<rank>{0, -1});

        return re;
    });
}

template <typename field_type>
inline auto div(field_type field) {
    static const int_t rank = field_type::rank;
    typedef typename field_type::value_type field_value_type;
    typedef typename field_value_type::value_type value_type;

    return make_lambda(field.shape(), [=](pointi<rank> idx) {
        auto nxx = field(idx + point2i{1, 0})[0] - field(idx + point2i{-1, 0})[0];
        auto nyy = field(idx + point2i{0, 1})[1] - field(idx + point2i{0, -1})[1];
        return nxx + nyy;
    });
}

template <typename image_type>
auto laplace(image_type image) {
    using value_type = typename image_type::value_type;
    static const int_t rank = image_type::rank;
    return make_lambda(image.shape(), [=](pointi<rank> idx) {
        auto tmp = -4 * image(idx);
        tmp = tmp + image(idx + pointi<2>{-1, 0});
        tmp = tmp + image(idx + pointi<2>{1, 0});
        tmp = tmp + image(idx + pointi<2>{0, -1});
        tmp = tmp + image(idx + pointi<2>{0, 1});
        return tmp;
    });
}

}  // namespace view
}  // namespace matazure

template <typename _ValueType, int_t _Rank>
inline auto normalize(const point<_ValueType, _Rank>& vec) {
    _ValueType sum_norm2 = 0;
    for (int_t i = 0; i < _Rank; ++i) {
        sum_norm2 += vec[i] * vec[i];
    }

    _ValueType scale = std::sqrt(sum_norm2);

    point<_ValueType, _Rank> re;
    for (int_t i = 0; i < _Rank; ++i) {
        re[i] = vec[i] / (scale + 1e-10);
    }

    return re;
}

template <typename image_type>
inline image_type make_border_copy(image_type img, int_t border) {
    //
    static const int_t rank = image_type::rank;
    pointi<rank> padding;
    fill(padding, border);

    image_type img_border(img.shape() + padding * 2);
    copy(img, view::crop(img_border, padding, img.shape()));

    auto shape = img_border.shape();

    // top
    for_index(point2i{0, 0}, point2i{shape[0], border},
              [=](point2i idx) { img_border(idx) = img_border(idx[0], border); });

    // bottom
    for_index(point2i{0, shape[1] - border}, shape,
              [=](point2i idx) { img_border(idx) = img_border(idx[0], shape[1] - 1 - border); });

    // left
    for_index(point2i{0, 0}, point2i{border, shape[1]},
              [=](point2i idx) { img_border(idx) = img_border(border, idx[1]); });

    // right
    for_index(point2i{shape[0] - border, 0}, shape,
              [=](point2i idx) { img_border(idx) = img_border(shape[0] - 1 - border, idx[1]); });

    return img_border;
}

template <typename image_type>
inline auto gradient(image_type image) {
    auto image_with_border = make_border_copy(image, 1);
    return view::gradient(view::crop(image_with_border, point2i{1, 1}, image.shape())).persist();
}

template <typename field_type>
inline auto div(field_type field) {
    auto field_with_border = make_border_copy(field, 1);
    return view::div(view::crop(field_with_border, point2i{1, 1}, field.shape())).persist();
}

template <typename image_type>
inline auto laplace(image_type image) {
    auto image_with_border = make_border_copy(image, 1);
    return view::laplace(view::crop(image_with_border, point2i{1, 1}, image.shape())).persist();
}

template <typename image_type>
inline void neumann_bound_conf(image_type image) {
    // clang-format off
    auto end = image.shape() - 1;    
    //border
    copy(view::slice<0>(image, 2), view::slice<0>(image, 0));
    copy(view::slice<0>(image, end[0] - 2), view::slice<0>(image, end[0]));
    copy(view::slice<1>(image, 2), view::slice<1>(image, 0));
    copy(view::slice<1>(image, end[1] - 2), view::slice<1>(image, end[1]));

    //corner
    image(0,      0)        = image(2,          2);
    image(end[0], 0)        = image(end[0] - 2, 2);
    image(0,      end[1])   = image(2,          end[1] - 2);
    image(end[0], end[1])   = image(end[0] - 2, end[1] - 2);
    // clang-format on
}

template <typename image_type>
image_type drlse_edge(image_type mat_phi0, image_type mat_g, float lambda, float mu, float alpha,
                      float epsilon, float timestep, int counter = 10) {
    typedef typename image_type::value_type value_type;
    static const size_t rank = image_type::rank;

    for_each(mat_g.shape() == mat_phi0.shape(),
             [](bool e) { MATAZURE_ASSERT(e, "shape must be matched"); });

    auto mat_g_grad = gradient(mat_g);
    auto mat_phi = mem_clone(mat_phi0, typename image_type::memory_type{});

    write_raw_data("mat_g.raw", mat_g);

    while (counter--) {
        // compute the curvature
        neumann_bound_conf(mat_phi);
        auto mat_phi_grad = gradient(mat_phi);
        auto mat_normalized_gradient_phi =
            view::map(mat_phi_grad, [](auto p) { return normalize(p); }).persist();
        auto mat_curvature = div(mat_normalized_gradient_phi);

        write_raw_data("mat_phi_curv.raw", mat_curvature);

        auto mat_dist_term = laplace(mat_phi) - mat_curvature;

        write_raw_data("mat_dist.raw", mat_dist_term.persist());

        auto mat_dirac_phi = view::map(mat_phi, [sigma = epsilon](value_type x) {
            if (std::abs(x) > sigma) return 0.0f;
            return (1.0f / (2 * sigma)) * (1.0f + std::cos(3.1415926f * x / sigma));
        });

        auto mat_area_term = mat_dirac_phi * mat_g;
        auto mat_temp = mat_g_grad * mat_phi_grad;
        auto mat_temp_sum = view::map(mat_temp, [](auto x) {
            auto tmp = zero<value_type>::value();
            for (int_t i = 0; i < x.size(); ++i) {
                tmp = tmp + x[i];
            }
            return tmp;
        });
        auto mat_edge_term = mat_dirac_phi * mat_temp_sum + mat_area_term * mat_curvature;
        auto mat_phi_old = mem_clone(mat_phi, typename image_type::memory_type{});
        auto mat_phi_next = mat_phi_old + timestep * (mu * mat_dist_term + lambda * mat_edge_term +
                                                      alpha * mat_area_term);
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