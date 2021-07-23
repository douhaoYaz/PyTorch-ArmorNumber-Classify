from torchvision import datasets, transforms
import os

data_dir = "./2021_7_23工业相机获取装甲板号码数据集"

# 单通道时
means = [0]
stdevs = [0]
# # 多通道时
# means = [0, 0, 0]
# stdevs = [0, 0, 0]

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(48),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=1),  # 彩色图像转灰度图像num_output_channels默认1
        transforms.ToTensor()
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(48),
        transforms.CenterCrop(48),
        transforms.Grayscale(num_output_channels=1),  # 彩色图像转灰度图像num_output_channels默认1
        transforms.ToTensor()
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_dataset = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

num_images = len(image_dataset['train'])
# # 多通道时
# for image in image_dataset['train']:
#     img = image[0]
#     for i in range(3):
#         means[i] += img[i, :, :].mean()
#         stdevs[i] += img[i, :, :].std()

# 单通道时
for image in image_dataset['train']:
    img = image[0]
    means[0] += img.mean()
    stdevs[0] += img.std()

means[0] /= num_images
stdevs[0] /= num_images
print('means: ', means[0])
print('stdves: ', stdevs[0])
# 2021_7_23工业相机获取装甲板号码数据集:  means: 0.2105   stdves: 0.2829