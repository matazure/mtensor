#include "sample_level_set.hpp"
#include <chrono>
#include "image_helper.hpp"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "sample_level_set input_image" << std::endl;
        return -1;
    }

    tensor<byte, 2> img_gray = read_gray_image(argv[1]);
    tensor<float, 2> img_float = view::cast<float>(img_gray).persist();
    auto img_grad = gradient(img_float);
    auto mat_g = view::map(img_grad,
                           [](point<float, 2> grad) -> float {
                               return 1.0f / (1.0f + grad[0] * grad[0] + grad[1] * grad[1]);
                           })
                     .persist();

    tensor<float, 2> mat_phi0(img_float.shape());
    float c0 = 2.0f;
    fill(mat_phi0, c0);
    fill(view::slice(mat_phi0, point2i{20, 25}, point2i{10, 10}), -c0);
    fill(view::slice(mat_phi0, point2i{40, 25}, point2i{10, 10}), -c0);

    float lambda = 5;
    float alfa = -3;
    float epsilon = 1.5f;
    float timestep = 1;
    float mu = 0.2 / timestep;

    auto t0 = std::chrono::high_resolution_clock::now();

    auto mat_phi = drlse_edge(mat_phi0, mat_g, lambda, mu, alfa, epsilon, timestep, 100);

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "cost time " << (t1 - t0).count() << " ns" << std::endl;

    tensor<byte, 2> img_phi(mat_phi.shape());
    transform(mat_phi, img_phi, [](float v) {
        return static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, -v * 100.f)));
    });

    write_gray_png("phi.png", img_phi);

    return 0;
}

/*

%  This Matlab code demonstrates an edge-based active contour model as an application of
%  the Distance Regularized Level Set Evolution (DRLSE) formulation in the following paper:
%
%  C. Li, C. Xu, C. Gui, M. D. Fox, "Distance Regularized Level Set Evolution and Its Application to
Image Segmentation", %     IEEE Trans. Image Processing, vol. 19 (12), pp. 3243-3254, 2010.
%
% Author: Chunming Li, all rights reserved
% E-mail: lchunming@gmail.com
%         li_chunming@hotmail.com
% URL:  http://www.engr.uconn.edu/~cmli/

clear all;
close all;

Img=imread('gourd.bmp');
Img=single(Img(:,:,1));
%% parameter setting
timestep=1;  % time step
mu=0.2/timestep;  % coefficient of the distance regularization term R(phi)
iter_inner=5;
iter_outer=20;
lambda=5; % coefficient of the weighted length term L(phi)
alfa=-3;  % coefficient of the weighted area term A(phi)
epsilon=1.5; % papramater that specifies the width of the DiracDelta function

sigma=.8;    % scale parameter in Gaussian kernel
G=fspecial('gaussian',15,sigma); % Caussian kernel
%Img_smooth=conv2(Img,G,'same');  % smooth image by Gaussiin convolution
Img_smooth = Img;
[Ix,Iy]=gradient(Img_smooth);
f=Ix.^2+Iy.^2;
g=1./(1+f);  % edge indicator function.

% initialize LSF as binary step function
c0=2;
initialLSF = c0*ones(size(Img));
% generate the initial region R0 as two rectangles
initialLSF(25:35,20:25)=-c0;
initialLSF(25:35,40:50)=-c0;
phi=initialLSF;

figure(1);
mesh(-phi);   % for a better view, the LSF is displayed upside down
hold on;  contour(phi, [0,0], 'r','LineWidth',2);
title('Initial level set function');
view([-80 35]);

figure(2);
imagesc(Img,[0, 255]); axis off; axis equal; colormap(gray); hold on;  contour(phi, [0,0], 'r');
title('Initial zero level contour');
pause(0.5);

potential=1;
if potential ==1
    potentialFunction = 'single-well';  % use single well potential p1(s)=0.5*(s-1)^2, which is good
for region-based model elseif potential == 2 potentialFunction = 'double-well';  % use double-well
potential in Eq. (16), which is good for both edge and region based models else potentialFunction =
'double-well';  % default choice of potential function end

% start level set evolution
for n=1:iter_outer
    phi = drlse_edge(phi, g, lambda, mu, alfa, epsilon, timestep, iter_inner, potentialFunction);
    if mod(n,2)==0
        figure(2);
        imagesc(Img,[0, 255]); axis off; axis equal; colormap(gray); hold on;  contour(phi, [0,0],
'r'); end end

% refine the zero level contour by further level set evolution with alfa=0
alfa=0;
iter_refine = 10;
phi = drlse_edge(phi, g, lambda, mu, alfa, epsilon, timestep, iter_inner, potentialFunction);

finalLSF=phi;
figure(2);
imagesc(Img,[0, 255]); axis off; axis equal; colormap(gray); hold on;  contour(phi, [0,0], 'r');
hold on;  contour(phi, [0,0], 'r');
str=['Final zero level contour, ', num2str(iter_outer*iter_inner+iter_refine), ' iterations'];
title(str);

figure;
mesh(-finalLSF); % for a better view, the LSF is displayed upside down
hold on;  contour(phi, [0,0], 'r','LineWidth',2);
view([-80 35]);
str=['Final level set function, ', num2str(iter_outer*iter_inner+iter_refine), ' iterations'];
title(str);
axis on;
[nrow, ncol]=size(Img);
axis([1 ncol 1 nrow -5 5]);
set(gca,'ZTick',[-3:1:3]);
set(gca,'FontSize',14)



*/