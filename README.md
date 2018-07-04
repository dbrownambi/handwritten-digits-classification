### This project implements machine learning algorithms to classify handwritten digits.
The complete project report can be found [here](https://github.com/dbrownambi/mnist-classification/blob/master/Project%20Report.pdf). The dataset used for this project is the **MNIST** database of handwritten digits, available from this [page](http://yann.lecun.com/exdb/mnist/). It has a training set of 60,000 examples, and a test set of 10,000 examples: 

![alt text](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png "MNIST")

The goal of this project is not to achieve the state of the art performance, rather to learn and compare the performance accuracy of 7 different machine learning algorithms namely:

1. [Linear Classifier](https://github.com/dbrownambi/mnist-classification/blob/master/models/Linear_SGD.py)
2. [K-Nearest Neighbor Classifier](https://github.com/dbrownambi/mnist-classification/blob/master/models/K-NN.py)
3. [Radial Basis Function Neural Network](https://github.com/dbrownambi/mnist-classification/blob/master/models/Radial%20BF.py)
4. [Fully Connected Multilayer Neural Network with One-Hidden Layer](https://github.com/dbrownambi/mnist-classification/blob/master/models/MLP-1%20hidden%20layer.py)
5. [Fully Connected Multilayer Neural Network with Two-Hidden Layers](https://github.com/dbrownambi/mnist-classification/blob/master/models/MLP-2%20hidden%20layer.py)
6. [Fully Connected Multilayer Neural Network with Three-Hidden Layers](https://github.com/dbrownambi/mnist-classification/blob/master/models/MLP-3%20hidden%20layer.py)
7. [Convolutional Neural Network](https://github.com/dbrownambi/mnist-classification/blob/master/models/CNN.py)

Although the solution isn't optimized for high accuracy, the results are quite good. Table below shows some results in comparison of the above-mentioned models:

![alt text](https://github.com/dbrownambi/mnist-classification/blob/master/Result.jpg "Result")
