# Hide and Seek : Weakly-supervised Object Localization

In this project a convolutional neural network is trained to classify and localize as well the objects using the global average pooling layer. CIFAR-10 dataset is used for the purpose. For more details about the main idea, refer [this](https://arxiv.org/pdf/1704.04232.pdf).

#### Preparing the dataset
1. The dataset is downloaded from [here](https://pjreddie.com/projects/cifar-10-dataset-mirror/)
2. Extract the files and then you will get the following file structure :
```
|-- cifar
|	|-- train
|	|	|-- airplane_1.jpg
|	|	|-- horse_1.jpg
|	|	|-- ...(all images mixed)
|	|-- test
|	|	|-- dog_1.jpg
|	|	|-- ship_1.jpg
|	|	|-- ...(all images mixed)
```
3. Create new folders named after the image classes in cifar-10 dataset i.e. Airplane, Automobile, cat,....., truck and copy the respectvie images from train folder and test folder to these folders accordingly.
>***Note** : There are some images which are different but have the same name in the given dataset.*
4. Paste the `renamer.sh` in each of the folders and edit the line number 5 in it for the respective class. This will rename all the images in it appropriately.
```
|-- cifar
|	|-- train
|	|-- test
|	|-- airplane
|	|	|-- ...(all airplane images)
|	|	|-- ...renamer.sh
|	|-- automobile
|	|	|-- ...(all automobile images)
|	|	|-- ...renamer.sh
|	|-- ...(rest of the classes)
|	|-- truck
|	|	|-- ...(all truck images)
|	|	|-- ...renamer.sh
```
5. Now separate the train and test sets.
6. After following steps 3, 4 and 5 the directory structure should look like this :
```
|-- cifar
|	|-- train
│	│	|-- Airplane
|	|	|	|-- 1.jpg
|	|	|	|-- ...
|	|	|	|-- n.jpg 
|	|	|-- ...
|	|	|-- Truck
|	|	|	|-- 1.jpg
|	|	|	|-- ...
|	|	|	|-- n.jpg
│	|-- test
│	│	|-- Airplane
|	|	|	|-- n+1.jpg
|	|	|	|-- same as the "train" folder.
```
7. Now the data is ready for the use.

#### Training models
All the models inside the **models** folder are trained on **colab** using different network structures and you can load them using keras and find its summary. The training file for colab is [cifar10_colab_train.ipynb](https://github.com/AtharvaKalsekar/Hide_and_Seek/blob/master/cifar10_colab_train.ipynb "cifar10_colab_train.ipynb") . Also [trainer.py](https://github.com/AtharvaKalsekar/Hide_and_Seek/blob/master/trainer.py "trainer.py") can be used for training on local machine.

#### Testing models
The models can be tested and examined using [testing_model.py](https://github.com/AtharvaKalsekar/Hide_and_Seek/blob/master/testing_model.py "testing_model.py"). This results in class prediction and localization of the test image.

#### Results
The classification accuracy is not so good. It has a validation accuracy of 78.32%. (Better classification can be obtained if more data is available).
The localization has good accuracy.

| | | |
|:-------------------------:|:-------------------------:|:-------------------------:|
|<img src="https://github.com/AtharvaKalsekar/Hide_and_Seek/blob/master/test_image_results/airplane_heatmap.jpg">  airplane heatmap |  <img src="https://github.com/AtharvaKalsekar/Hide_and_Seek/blob/master/test_image_results/bird_heatmap.jpg"> bird heatmap|<img src="https://github.com/AtharvaKalsekar/Hide_and_Seek/blob/master/test_image_results/ship_heatmap.jpg"> ship heatmap|
|<img src="https://github.com/AtharvaKalsekar/Hide_and_Seek/blob/master/test_image_results/airplane_bbox.jpg"> airplane bbox |  <img src="https://github.com/AtharvaKalsekar/Hide_and_Seek/blob/master/test_image_results/bird_bbox.jpg"> bird bbox |<img src="https://github.com/AtharvaKalsekar/Hide_and_Seek/blob/master/test_image_results/ship_bbox.jpg"> ship bbox |
