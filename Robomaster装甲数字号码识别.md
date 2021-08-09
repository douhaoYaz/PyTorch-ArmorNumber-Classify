# Robomaster装甲数字号码识别



## 开发过程

* 看深度学习与PyTorch入门实战
* 根据在Github找的开源代码，用装甲板数据集训练Resnet模型
* 把Resnet模型从.pth转换成.pt格式，并尝试使用libtorch把模型部署在Ubuntu QT5 环境C++代码上
* 把Resnet模型从.pth转换成.onnx格式，并尝试把.onnx格式的Resnet模型部署在Ubuntu QT5 环境C++代码上
* 在Ubuntu上用OpenCV进行数据预处理
* 更换模型为Lenet-5以降低数字号码识别处理时间





## 训练模型

参考Github开源的PyTorch训练模型项目，使用48×48的装甲板0到5号号码图片作为数据集训练resnet18模型

写该markdown时整理参考了的博客时，发现一篇blog是基于这个Github开源改的，并且有注释：[C++利用opencv调用pytorch训练好的分类模型](https://blog.csdn.net/qq_30263737/article/details/114287291)





## libtorch部署模型到Ubuntu QT5 环境C++代码

模型训练好了，接下来需要把模型部署到代码里

参考了一些CSDN的资料：

* [ubuntu16.04使用C++调用pytorch训练的模型，并使用opencv进行数据加载和预测](https://blog.csdn.net/qq_36852276/article/details/106343051)
* [Ubuntu下C++调用pytorch训练好模型--利用libtorch](https://blog.csdn.net/qq_36481821/article/details/107504333?ops_request_misc=&request_id=&biz_id=102&utm_term=PyTorch%20libtorch&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-8-.first_rank_v2_pc_rank_v29&spm=1018.2226.3001.4187)

先在PyTorch官网下载libtorch

然后需要把libtorch导入QT的.pro文件里配置

参考了CSDN文章并综合了自己的视觉代码来编写.pro文件，但是编译时QT总是报奇怪的错，怎样也无法通过编译

很大可能是因为libtorch只能和OpenCV 3.4.0兼容（一些blog和知乎里提到）

也有可能是因为没有安装cuda（虽然说libtorch有cpu版和cuda版，但两者都尝试过仍然无法通过编译）

遇到的困难是：我的电脑给Ubuntu系统分配的存储空间不够了，无法再安装cuda，除非重装重新分配空间，但是临近比赛，时间不允许

另外是实验室的Intel NUC是没有GPU的，装了cuda也没用，而且NUC已经安装了OpenCV 4.4.0，这段时间经常需要测试，无法承受重装OpenCV 3.4.0的风险

于是只好暂时放弃用libtorch部署模型的方案，正好看到blog介绍到，除了用libtorch部署外，还可以把模型格式转换为onnx格式，OpenCV支持onnx格式的模型



## PyTorch将.pth格式转换成.onnx格式

参考资料：

* PyTorch官网 > Tutorials > [(optional) Exporting a Model from PyTorch to ONNX and Running it using ONNX Runtime](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
* PyTorch中文教程 > [将模型从 PyTorch 导出到 ONNX 并使用 ONNX 运行时运行它（可选）](https://pytorch.apachecn.org/docs/1.7/40.html)  （上面的中文翻译）
* 还有一些CSDN文章，不过都不如官网的好

编写pth2onnx.py（因为官网的例子不太适用于我的情况，我这里已经训练好一个模型了，所以下面的版本是综合了官网例子和几篇CSDN博客写的）

```python
# torchvision里预先定义好的模型resnet18
model=torchvision.models.resnet18(pretrained=True)
# 初始化模型结构参数
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 6)
# 加载之前训练好的renset18模型参数
model.load_state_dict(torch.load("resnet18_ArmorNum.pth"))
# 设为推理模式
model.eval()

batch_size = 1
# 随机生成一个输入
example = torch.rand(batch_size, 3, 48, 48, requires_grad=True)
# 导出onnx模型
export_onnx_file = "resnet18v8.onnx"
torch.onnx.export(model,
                  example,
                  export_onnx_file,
                  export_params=True,
                  # verbose=True,
                  opset_version=10,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output']
                 )
```





## 部署.onnx格式Resnet模型到Ubuntu QT5 环境C++代码

将resnet18.onnx模型文件copy到Intel NUC上

因为直接把模型部署到Robomaster视觉代码上面运行需要连接摄像头并使用装甲板，为避免测试成本太高，于是编写ONNX_Test.cpp代码

在ONNX_Test代码中，可以加载48×48的装甲板图片，并加载resnet18_ArmorNum.onnx模型来初始化OpenCV中dnn模块的Net类，使用net类可以在C++实现前向传播，从而得到预测结果，参考了以下博客

* [OpenCV加载ONNX模型并推理](https://blog.csdn.net/MrTianShu/article/details/118219547)
* [Pytorch训练的分割网络模型在OpenCV4.0/C++上部署](https://blog.csdn.net/ppCuda/article/details/103393679)

遗憾的是预测结果并不准确，猜测是以下几个地方存在问题：

* 模型从.pth格式转换到.onnx格式后可能网络结构会产生变化，导致相同的输入在.pth和.onnx上会得到不同的输出
* 在Ubuntu QT5 C++环境，使用OpenCV的dnn模块Net类的forward前向传播，与Windows Pycharm PyTorch环境的前向传播，两者会得到不同的输出
* 使用cuda训练的模型，在没有cuda的环境下跑是否会对结果有影响
* 在PyTorch和在C++对输入数据的处理不同

根据以上几点寻找问题所在





## 寻找在Ubuntu OpenCV C++环境下.onnx格式模型得不到正确预测的原因

对于同一装甲板图像数据，在Windows PyTorch下使用resnet18_ArmorNum.pth模型能够得到正确预测值

但在Ubuntu OpenCV C++下使用resnet18_ArmorNum.onnx模型不能得到正确预测值

### 尝试调整.pth转.onnx格式的参数设置

猜测可能是.pth格式转.onnx格式的转换操作参数设置不正确

参考了CSDN上.pth格式转.onnx格式的操作，对pth2onnx.py文件进行修改

* [Pytorch训练的分割网络模型在OpenCV4.0/C++上部署](https://blog.csdn.net/ppCuda/article/details/103393679)
* [pytorch转换为onnx](https://blog.csdn.net/yangdashi888/article/details/104198844)
* [pytorch转onnx模型并进行推理](https://blog.csdn.net/qq_36202348/article/details/108984612)

文章修改版本命名为resnet18v2.onnx、resnet18v3.onnx等，并copy到NUC上，在ONNX_Test代码中加载修改后的模型并查看预测值

遗憾的是，参考CSDN上转换为.onnx格式的操作对pth2onnx.py的修改，无一能在ONNX_Test上得到正确输出

于是打算直接测试比较两格式模型

### 测试resnet18_ArmorNum.pth模型及其.onnx格式 在相同输入前提下是否能得到相同的预测输出

对于模型从.pth格式转换到.onnx格式后网络结构是否会产生变化的问题，我们需要验证相同的输入是否会在.pth和.onnx上得到不同输出

只需将原先的resnet18_ArmorNum.pth转换为resnet18_ArmorNum.onnx，并使用相同的值作为输入，比较两者输出即可判断问题是否出在模型转换上

因此继续编写pth2onnx.py文件，此时代码中已经完成了.pth转换为.onnx并导出，在此基础上，随机生成一个tensor作为输入，并经过.pth格式的模型得到预测输出

剩下的就是用.onnx格式的模型输出预测，但问题是如何在Pytorch上用.onnx格式的模型进行前向传播

此时想起PyTorch官网将.pth转换成.onnx的教程：

* PyTorch官网 > Tutorials > [(optional) Exporting a Model from PyTorch to ONNX and Running it using ONNX Runtime](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)

之前只参考了将.pth转换成.onnx的部分，教程中还有验证onnx模型结构并使用ONNXRuntime来运行模型输出预测

于是参考PyTorch官方和CSDN博客 [Pytorch模型转成onnx并验证结果(排坑经验帖)](https://muzhan.blog.csdn.net/article/details/112642436) 编写验证并运行onnx模型的代码

然后获得onnx模型的预测输出，比较.pth和.onnx的输出，发现两者结果一致

基本得出结论：问题不是出在模型转换上

## 验证数据预处理导致预测结果不一致 并研究预处理对数据造成的改变

在NUC上进行Ubuntu QT5 OpenCV C++的ONNX_Test测试时，观察到输出的预测其中一位元素的值高达200，因为已基本排除是模型转换的问题，于是就怀疑问题出在输入的数据上。需要验证以下两点：

* 在Windows Pytorch下的输入数据处理是否与在Ubuntu OpenCV C++上的一致
* .pth格式和.onnx格式的模型的输入数据类型是否不一样

检查代码发现，PyTorch下对输入的图片进行了transforms处理，而OpenCV C++上并没有对输入的图片数据进行处理，也许这就是两者处理的预测结果不一致的原因

```python
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
```

为了验证上述猜想，于是编写compare_onnx_pth.py文件

* 加载.pth和.onnx格式模型
* 加载图片用作输入
* 对输入.pth和.onnx格式模型的图片进行一样的数据预处理
* 观察.pth和.onnx格式模型的预测输出是否一致

经过测试，发现同一图像经过相同处理后，.pth和.onnx模型输出的预测一致

破案! 是数据预处理的问题，在OpenCV C++上没有进行图像的数据处理

### 研究以上预处理对数据产生的影响并将其作为在OpenCV C++上修改的根据

为了在Ubuntu上用OpenCV完成 与在Windows上用PyTorch所作一样的数据预处理，需要研究以上预处理是如何改变数据的

在predict.py中，逐个测试transforms.Compose中各项对图像数据的处理

```python
transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

测试发现如果输入图片的size和input_size一样的话（我们装甲板图片的size是48×48，且input_size也设为了48）transform.Resize和transform.CenterCrop处理后数据不产生变化，而transforms.ToTensor和transforms.Normalize这两个transforms就是导致Windows PyTorch下.pth模型和Ubuntu OpenCV C++下.onnx模型的预测结果不同的原因了



查阅PyTorch官方文档，ToTensor会把PIL格式的Image和numpy.ndarray格式 (H x W x C) 的图片数据转换为tensor(C x H x W)

**至关重要的是**，PIL格式和numpy.ndarray格式的值的范围是[0, 255]，而tensor格式的值的范围是[0.0, 1.0]

因此ToTensor方法会把图片的像素值的范围从[0, 255]缩放到[0.0, 1.0]，而使用python的Image.open方法打开的图片是PIL格式的

> Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] 

并根据 [pytorch中transforms.ToTensor()函数解析](https://blog.csdn.net/wuqingshan2010/article/details/110133046?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522162650612116780366575233%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=162650612116780366575233&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_v2~rank_v29-6-110133046.first_rank_v2_pc_rank_v29&utm_term=pytorch+transform+toTensor%E4%B9%8B%E5%90%8E%E5%8F%98%E5%9B%9Enumpy&spm=1018.2226.3001.4187) 和 [torchvision.transforms.ToTensor（细节）对应caffe的转换](https://blog.csdn.net/qq_22764813/article/details/103404462) 得知缩放的方法是每个像素值除以255



查阅PyTorch官方文档，Normalize会根据输入的mean均值和std方差对tensor进行归一化，具体就是对各个channel的每个值减去对应channel的mean均值再除以std方差

> Normalize a tensor image with mean and standard deviation. This transform does not support PIL Image. Given mean: `(mean[1],...,mean[n])` and std: `(std[1],..,std[n])` for `n` channels, this transform will normalize each channel of the input `torch.*Tensor` i.e., `output[channel] = (input[channel] - mean[channel]) / std[channel]`



输入的PIL格式图片经过transform之后（其中的ToTensor方法使其变为tensor类型），还要unsqueeze进行维度扩张，扩张成(batch, channel, H, W)

最终输入到.pth和.onnx模型，两者得到的预测结果一致（此处一定注意.pth模型的输入类型是tensor，而.onnx模型的输入类型是numpy.ndarray格式）

终于证实了用同一装甲板号码图片作为输入，在Windows PyTorch与Ubuntu OpenCV C++会得到不同预测结果的原因是，数据的预处理





## 在Ubuntu上用OpenCV进行数据预处理

根据对PyTorch中transform操作的研究，只要在Ubuntu OpenCV C++上对输入数据用同样的缩放以及归一化，就能输入到.onnx的神经网络模型中，得到与在Windows PyTorch下预测同样的结果

在参考了CSDN一篇blog对图像进行归一化处理后，了解到其中使用到的OpenCV的convertTo函数，可以批量地对数据进行缩放和归一化，符合我们对数据处理的要求

本来考虑用eigen库进行矩阵运算，但是发现用OpenCV的convertTo函数只要4行代码就能实现数据的缩放和归一化了

```C++
//time spend
double t_end, t_begin;

//Initialize Neurel Network
cv::dnn::Net net = cv::dnn::readNetFromONNX("/home/douhao/resnet18.onnx");

//Load image
cv::Mat image = cv::imread("/home/douhao/ONNX模型及图片测试集/2/985.png");
if(image.empty())
    std::cout << "load image failed\n" << std::endl;

//Set mean and std
std::vector<float> mean_value = {0.485, 0.456, 0.406};
std::vector<float> std_value = {0.229, 0.224, 0.225};

t_begin = cv::getTickCount();

//ImageProcess
cv::Mat dst;
std::vector<cv::Mat> bgrChannels(3);
cv::split(image, bgrChannels);
for(auto i = 0; i < bgrChannels.size(); i++){
    bgrChannels[i].convertTo(bgrChannels[i], CV_32FC1, 1.0 / 255, -mean_value[i]);
    bgrChannels[i].convertTo(bgrChannels[i], CV_32FC1, 1.0 / std_value[i], 0.0);
}
cv::merge(bgrChannels, dst);

//forward
cv::Mat blob = cv::dnn::blobFromImage(dst);
net.setInput(blob);
cv::Mat pred = net.forward();
std::cout << "pred:\n" << pred << std::endl;

t_end = cv::getTickCount();
std::cout << "spend " << ((t_end - t_begin)/ cv::getTickFrequency() *1000) << " ms"  << std::endl;
```





## 更换模型为Lenet-5以降低处理时间

使用Resnet18进行装甲板数字号码识别的处理时间太长，经测试达到40+ms，完全不能适应我们Robomaster的比赛要求

因此决定把模型从18层的Resnet18更换成8层的Lenet-5



为此编写Lenet5.py、Lenet5_train_val_ArmorNum.py、predict_Lenet5_for_ArmorNum.py、pth2onnx_Lenet5.py

Lenet5.py的编写参考了新加坡国立大学龙良曲老师的深度学习与PyTorch实战介绍的Lenet5

