# Tensor
tensor是matazure下的一个多维数组基础库, 具有以下的优点
* 同时支持C++和CUDA
* 具备泛型(多类型,多维度)tensor
* 具备函数式lambda_tensor
* 向量化的tensor操作和运算
* 延迟计算
* 内存回收
* Header Only

## 如何使用
将tensor根目录加入到目标项目的头文件路径即可

## 实例

## 生成项目
* Windows  

* Ubuntu 16

## 其他
安装[ImageMagic](http://www.imagemagick.org/)工具

将图像转换为二进制raw data
```
convert   lena_rgb888_512x512.jpg  -depth 8 rgb:lena_rgb888_512x512.raw_data
convert   lena_rgb888_512x512.jpg  -depth 8 gray:lena_gray8_512x512.raw_data
```
将二进制raw data转换为图像
```
convert  -size 512x512 -depth 8 rgb:lena_rgb888_512x512.raw_data lena_rgb888_512x512.jpg
convert  -size 512x512 -depth 8 gray:lena_gray8_512x512.raw_data lena_gray8_512x512.jpg
```
