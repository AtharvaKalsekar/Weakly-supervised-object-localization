# Hide and Seek : Weakly-supervised Object Localization

In this project a convolutional neural network is trained to classify and localize as well the objects using the global average pooling layer. CIFAR-10 dataset is used for the purpose. For more details about the main idea, refer [this](https://arxiv.org/pdf/1704.04232.pdf).

#### Preparing the dataset
1. The dataset is downloaded from [here](https://pjreddie.com/projects/cifar-10-dataset-mirror/)
2. Extract the files and then you will get the following file structure :
    `cifar/`
     `|------> train/...(images)`
    `|------> test/...(images)`
3. Create new folders named after the image classes in cifar-10 dataset i.e. Airplane, Automobile, cat,....., truck and copy the respectvie images from train folder to these folders accordingly.
4. Repeat the same process for test folder as well by creating new class folders again.
5. After following steps 3 and 4 the directory strucutre should look like this :
    `cifar/`
    `|---->train/Airplane/...(airplane images).`
    `|---->train/Automobile/...(automobile images).`
    `.`
    `.`
    `.`
    `|---->train/truck/...(truck images).`
    `|---->test/Airplane/...(airplane images).`
    `|---->test/Automobile/...(automobile images).`
    `.`
    `.`
    `.`
    `|---->test/truck/...(truck images).`

*Note : There are some images which are different but have the same name in the given dataset.*

7. Run `renamer.sh` to rename the images in dataset appropriately.
8. Now the data is ready for the use.
