# pytorch


TorchVision is a package which consists of popular datasets, models and computer vision utilities such as transforms, display and writing videos/images, etc.

Torchvision consists of the following classes

* Datasets
* Transforms
* Models
* Utils
* IO
* Ops

torchvision.datasets.DATASET(root, train=True, transform=None, target_transform=None, download=False)


* DATASET is the name of the dataset, which can be MNIST, FashionMNIST, COCO etc. Get the full list here
* root is the folder that stores the dataset. Use this if you opt to download.
* train is a flag that specifies whether you should use the train data or test data
* download is a flag which is turned on when you want to download the data. Note that the data is not downloaded if it is already present in the root folder mentioned above.
* transform applies a series of image transforms on the input images. For example, cropping, resizing, etc.
* target_transform takes the target or labels and transforms it as required.
