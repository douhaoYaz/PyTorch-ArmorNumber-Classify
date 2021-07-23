# Robomaster装甲数字号码识别——使用单通道图像

前言：之前的装甲数字号码识别是基于三通道图像的，而因为在视觉代码上用单通道的图像分类器效率更高且更加方便，因此在之前基础上做了适当修改，使代码能适配处理单通道图像



## 用OpenCV将数据集的图像从三通道转换成单通道

编写3channels_to_singleChannel.cpp

使用了cv::glob函数把文件夹的文件读到vector中

然后使用cv::cvtColor将三通道图像转换为单通道图像

最后cv::imwrite将单通道图像保存



参考资料：[opencv3中的glob函数读取文件夹中数据](https://blog.csdn.net/qq_31261509/article/details/79460639?)



## 修改Lenet5网络结构

对Lenet5.py文件作修改

因为输入层从三通道变成了单通道

因此第一层卷积层Conv2d需要把第一个参数改为1

再修改下注释即可



## 修改训练模型代码

对Lenet5_train_val_ArmorNum.py作修改

由于数据集从三通道变成了单通道，因此用transforms对数据做的预处理也需要修改

首先需要在transforms.ToTensor()之前增加一项transforms.Grayscale(num_output_channel=1)

作用是将三通道图像转换为单通道图像（虽然上面已经用OpenCV将数据集转换成单通道了，但是不加这一项transforms则会报如下错误）

![image-20210723213601350](https://gitee.com/douhaoYaz/typora_pic/raw/master/Typora_pic/image-20210723213601350.png)

transforms.Normalize归一化的处理也要修改，只需输入单通道的均值和标准差

这时需要计算出数据集所有图像的均值和标准差，作为transforms.Normalize的输出（其实之前三通道的也可以，不过为了赶进度就直接用了ImageNet的均值和标准差）



参考资料：[pytorch中彩色图像（三通道）转灰度图像（单通道）](https://blog.csdn.net/qq_40821163/article/details/109436438)



## 计算数据集均值和标准差

编写calculate_means_stdevs.py

在对数据集进行与训练模型时同样的数据预处理后

对图像的每个通道调用.mean()和.std()函数计算均值和标准差

print打印出来，就可以手动输入到训练模型的transforms.Normalize数据预处理



参考资料：[Pytorch：计算图像数据集的均值和标准差](https://blog.csdn.net/DragonGirI/article/details/107578490)

