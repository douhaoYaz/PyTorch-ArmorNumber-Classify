# Robomaster装甲数字号码识别



## 开发流程

* 开发环境
* PyTorch编写神经网络
* 训练模型
* 检查模型（可选）
* 转换模型格式为onnx
* 检查onnx的模型
* 将onnx模型部署到OpenCV C++上





## 开发环境

训练模型：Windows Pycharm PyTorch

部署模型：Ubuntu QT5 OpenCV C++





## PyTorch编写神经网络

此处编写Lenet-5网络

因为Robomaster的视觉处理对速度要求比较高，模型结构越简单越好，前向传播所花时间也越短（前提是保证准确率，不过也不用特别高）

```python
# Lenet5.py

import torch
from torch import nn
from torch.nn import functional as F

class Lenet5(nn.Module):
    """
    for ArmorNum dataset
    """
    def __init__(self):
        # 调用类的初始化方法来初始化父类
        super(Lenet5, self).__init__()

        # 新建一个conv_unit变量
        # 用Sequential包含网络，可以方便地组织各种结构
        self.conv_unit = nn.Sequential(
            # 建立一个卷积层
            # x:[b, 3, 48, 48] => [b, 6, ?, ?] 大小size暂时未知，因为它与kernel_size、stride和padding有关，大概在32左右
            # 根据Yann LeCun的paper，第一个卷积层的输出是6个channels
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            # 在Yann LeCun的Lenet5 paper中，第二层是个Subsampling层，我们这里用pooling池化层
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            # 第三层，第二个卷积层
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            # 第四层，再来一个池化层
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            # 接下来是Full connection全连接层，需要用Flatten，将经过上面四层处理的输出原本是四维的转换成二维，因为Pytorch没有这个转换的类，所以不能在Sequential里完成Full connection,只能用view方法进行flatten
            # 因此建立两个unit，一个是这里的conv_unit，一个是下面的fc_unit
        )
        
        # 临时构造一个tmp数据，用于测试conv_unit到fc_unit的第一个Linear层的第一个参数是多少
        # [b, 3, 48, 48]
        tmp = torch.randn(2, 3, 48, 48)
        out = self.conv_unit(tmp)
        # [b, 16, 9, 9]
        print('conv out:', out.shape)
        
        
if __name__ == '__main__':
    net = Lenet5()
    
```

为了确认conv_unit到fc_unit的第一个linear层的第一个参数是多少，需要临时构造一个数据输入conv_unit运行，查看输出的shape

得到输出的shape后，继续编写fc_unit全连接层单元

```python
# Lenet5.py

import torch
from torch import nn
from torch.nn import functional as F

class Lenet5(nn.Module):
    """
    for ArmorNum dataset(3×48×48）
    """
    def __init__(self):
        # 调用类的初始化方法来初始化父类
        super(Lenet5, self).__init__()

        # 新建一个conv_unit变量
        # 用Sequential包含网络，可以方便地组织各种结构
        self.conv_unit = nn.Sequential(
            # 建立一个卷积层
            # x:[b, 3, 48, 48] => [b, 6, ?, ?] 大小size暂时未知，因为它与kernel_size、stride和padding有关，大概在32左右
            # 根据Yann LeCun的paper，第一个卷积层的输出是6个channels
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            # 在Yann LeCun的Lenet5 paper中，第二层是个Subsampling层，我们这里用pooling池化层
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            # 第三层，第二个卷积层
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            # 第四层，再来一个池化层
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            # 接下来是Full connection全连接层，需要用Flatten，将经过上面四层处理的输出原本是四维的转换成二维，因为Pytorch没有这个转换的类，所以不能在Sequential里完成Full connection,只能用view方法进行flatten
            # 因此建立两个unit，一个是这里的conv_unit，一个是下面的fc_unit
        )

        # Full connection unit，全连接层
        self.fc_unit = nn.Sequential(
            # 因为不知道conv_unit输出的shape是怎样的，因此下面nn.Linear的第一个参数需要通过下面的tmp和out测试conv_unit输出的shape来决定
            # 这里的16*5*5是已经通过测试得到的
            nn.Linear(16 * 9 * 9, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 6)
        )

        # # 临时构造一个tmp数据，用于测试conv_unit到fc_unit的第一个Linear层的第一个参数是多少
        # # [b, 3, 48, 48]
        # tmp = torch.randn(2, 3, 48, 48)
        # out = self.conv_unit(tmp)
        # # [b, 16, 9, 9]
        # print('conv out:', out.shape)

        # # 因为softmax()函数的输出不稳定，因此这里使用包含了softmax()的CrossEntropyLoss
        # self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        :param x: [b, 3, 48, 48]
        :return: logits
        """
        batchsz = x.size(0)
        # [b, 3, 48, 48] => [b, 16, 9, 9]
        x = self.conv_unit(x)
        # [b, 16, 9, 9] => [b, 16*9*9]
        x = x.view(batchsz, 16*9*9)     # 也可写成x = x.view(batchsz, -1)
        # [b, 16*9*9] => [b, 6]
        # 因为经过fc_unit全连接层之后，还要经过softmax()函数处理，这个在全连接层后softmax()前的输出就叫logits
        logits = self.fc_unit(x)

        return logits
```

这个Lenet5.py作为一个模块

接下来编写训练模型的代码，可以在里面调用Lenet5.py模块





## 训练模型

接下来需要训练并导出训练好的模型

为此编写Lenet5_train_val.py，代码比较长就不放在md文件里了

在原来代码的基础上，只需要修改以下地方来适配：

* data_dir：数据集的路径，必须包含train和val两个目录，代码会从该路径加载训练集和验证机
* output_path：模型文件输出的路径和名称
* model_name：训练网络名
* lr_rate：学习率
* num_classses：种类数目，必须和要调用的网络匹配
* batch_size：不能太小，也不能太大
* num_epoches：训练次数

如果想要调用其他模型，需要在main里修改

```python
model_ft = Lenet5()
```

修改Lenet5()为其他网络。不过记得先把那个网络的文件导入才能调用

同时根据需要调整input_size，也就是输入数据的(B, C, H, W)的后两维



运行Lenet5_train_val.py：

* 在Windows“开始”，打开Anaconda Prompt
* cd到Pycharm的该项目的目录
* 输入python Lenet5_train_val.py即可运行程序



## 检查模型（可选）

训练好模型后，可以加载一张图片作为输入来检查模型

```python
# predict_Lenet5_for_ArmorNum.py

import torchvision as tv
import torchvision.transforms as transforms
import torch
from PIL import Image
import torch.nn as nn
from Lenet5 import Lenet5

input_size = 48
names = ['0', '1', '2', '3', '4', '5']
def pridict():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=Lenet5()
    model.load_state_dict(torch.load("Lenet5_ArmorNum.pth"))
    model = model.to(device)
    model.eval()  # 推理模式

    # 获取测试图片，并行相应的处理
    img = Image.open('test.png')
    # 查看转换前的img的格式
    print("img shape before trransform:", img)
    transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    img = transform(img)
    # 查看转换后的img的格式
    print('img shape after transform:', img.shape)
    # print('img pixel after transform:', img)
    img = img.unsqueeze(0)
    # 查看unsqueeze之后的img的格式
    print('img shape after unsqueeze:', img.shape)
    img = img.to(device)


    with torch.no_grad():
        py = model(img)
    _, predicted = torch.max(py, 1)  # 获取分类结果
    classIndex_ = predicted[0]

    print('predict:', py)
    print('预测结果：', names[classIndex_])


if __name__ == '__main__':
    pridict()

```

同时可查看输入图片经过transform和unsqueeze前后的格式





## 转换模型格式为onnx并检查模型

参考资料：

* PyTorch官网 > Tutorials > [(optional) Exporting a Model from PyTorch to ONNX and Running it using ONNX Runtime](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
* PyTorch中文教程 > [将模型从 PyTorch 导出到 ONNX 并使用 ONNX 运行时运行它（可选）](https://pytorch.apachecn.org/docs/1.7/40.html)  （上面的中文翻译）
* 还有一些CSDN文章，不过都不如官网的好

编写pth2onnx.py（因为官网的例子不太适用于我的情况，我这里已经训练好一个模型了，所以下面的版本是综合了官网例子和几篇CSDN博客写的）

```python
# pth2onnx_Lenet5.py

import torch
import torchvision
import torch.nn as nn
import onnx
import onnxruntime
import numpy as np
from Lenet5 import Lenet5

#model = torchvision.models.resnet50(pretrained=True)
model=Lenet5()
model.load_state_dict(torch.load("Lenet5_ArmorNum.pth"))

model.eval()

batch_size = 1
# example = torch.rand(1, 3, 224, 224)
example = torch.rand(batch_size, 3, 48, 48, requires_grad=True)

# print output with the purpose of comparing pth and onnx
output_pth = model(example)
print('output_pth:', output_pth)

# --------------------------------
export_onnx_file = "Lenet5_v1.onnx"
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

# 使用ONNX的api检查ONNX模型
# 加载保存的模型并输出onnx.ModelProto结构
onnx_model = onnx.load("Lenet5_v1.onnx")
# 验证模型的结构并确认模型具有有效的架构
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession("Lenet5_v1.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name:to_numpy(example)}
ort_outputs = ort_session.run(None, ort_inputs)

print('output_onnx:', ort_outputs)

# compare ONNX Runtime and Pytorch results
np.testing.assert_allclose(to_numpy(output_pth), ort_outputs[0], rtol=1e-03, atol=1e-05)

print('Exported model has been tested with ONNXRuntime, and the result looks good!')
```





## 将onnx模型部署到OpenCV C++上

需要在Ubuntu OpenCV C++上对输入数据用和PyTorch中transform操作同样的缩放以及归一化

在参考了CSDN一篇blog对图像进行归一化处理后，了解到其中使用到的OpenCV的convertTo函数，可以批量地对数据进行缩放和归一化，符合我们对数据处理的要求

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



