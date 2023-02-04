# pytorch


TorchVision is a package which consists of popular datasets, models and computer vision utilities such as transforms, display and writing videos/images, etc.

Torchvision consists of the following classes

* Datasets
* Transforms
* Models
* Utils
* IO
* Ops

============================================================================================

torchvision.datasets.DATASET(root, train=True, transform=None, target_transform=None, download=False)


* DATASET is the name of the dataset, which can be MNIST, FashionMNIST, COCO etc. Get the full list here
* root is the folder that stores the dataset. Use this if you opt to download.
* train is a flag that specifies whether you should use the train data or test data
* download is a flag which is turned on when you want to download the data. Note that the data is not downloaded if it is already present in the root folder mentioned above.
* transform applies a series of image transforms on the input images. For example, cropping, resizing, etc.
* target_transform takes the target or labels and transforms it as required.

============================================================================================

torchvision.transforms.ToTensor - It takes in a PIL image of dimension [H X W X C] in the range [0,255] and converts it to a float Tensor of dimension [C X H X W] in the range [0,1].

torchvision.transforms.Compose - It chains many transforms together so that you can apply then all in one go.

============================================================================================

torchvision.models has many well-known models for computer vision tasks

eg:
Classification
Detection
Segmentation
Video Classification

model = torchvision.models.MODEL(pretrained=True)
Where,

MODEL is the name of the model such as AlexNet, ResNet etc. Check the full list of available models here [https://pytorch.org/vision/stable/models.html].

pretrained is the flag which specifies whether you want the model to be initialized with the pretrained weights of the model or not. If set to True, it will also download the weights file, when absent.

============================================================================================
