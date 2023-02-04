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

Utils
It has 2 nice functions which come in handy while dealing with images and publishing findings of your work.

1.Make grid of images for display

torchvision.utils.make_grid(tensor, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)


* tensor – 4D mini-batch Tensor of shape (B x C x H x W) or a list of images all of the same size.
* nrow – Number of images displayed in each row of the grid. The final grid size is (B / nrow, nrow). Default: 8.
* padding – amount of padding. Default: 2.
* normalize – If True, shift the image to the range (0, 1), by the min and max values specified by range. Default: False.
* range – tuple (min, max) where min and max are numbers, then these numbers are used to normalize the image. By default, min and max are computed from the tensor.
* scale_each – If True, scale each image in the batch of images separately rather than the (min, max) over all images. Default: False.
* pad_value – Value for the padded pixels. Default: 0.

2.Save Image

torchvision.utils.save_image(tensor, fp, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0, format=None)

If you provide a mini-batch to the above function, it saves them as a grid of images. The other arguments are similar to make_grid

============================================================================================
