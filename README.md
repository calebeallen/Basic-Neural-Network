# Basic-Neural-Network

This project is a basic neural network written from scratch in c++. I developed this project to learn c++ and understand machine learning.

The NeuralNetwork class constructor takes a learning rate, layer count, and nodes per layer array as parameters making it scalable and suitable for many basic machine learning applications. 

This model can be trained on many types of data. For simplicity, I generated training data using a simple function. The function generates an array of 30 values from a seed (a value 0-9). Then, a random decimal is added to each value. The indicies of the values are also offset by a random value. This ideally simulates data that would come from simple datasets such as the MNIST set containing hand written digits.

The following image shows the network's average cost and performance after 30k training iterations. The network confidently and correctly classifies the input seed.

![NN perf](https://github.com/calebeallen/Basic-Neural-Network/assets/147087056/32274558-9fe6-4e01-9b7a-5c745da7729b)
