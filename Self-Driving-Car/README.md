# Self Driving Car (End to End CNN/Dave-2)
![alt_text](http://www.techholic.co.kr/news/photo/201712/172710_136873_1941.jpg)

Refer the <a href="https://github.com/vj98/ML-Projects/blob/master/Self-Driving-Car/Self_driving_car.ipynb">Self Driving Car Notebook</a> for complete Information

1. Used convolutional neural networks (CNNs) to map the raw pixels from a front-facing camera to the steering commands for a self-driving car. This powerful end-to-end approach means that with minimum training data from humans, the system learns to steer, with or without lane markings, on both local roads and highways. The system can also operate in areas with unclear visual guidance such as parking lots or unpaved roads.
2.The system is trained to automatically learn the internal representations of necessary processing steps, such as detecting useful road features, with only the human steering angle as the training signal. We do not need to explicitly trained it to detect, for example, the outline of roads.
3. End-to-end learning leads to better performance and smaller systems. Better performance results because the internal components self-optimize to maximize overall system performance, instead of optimizing human-selected intermediate criteria, e. g., lane detection. Such criteria understandably are selected for ease of human interpretation which doesn’t automatically guarantee maximum system performance. Smaller networks are possible because the system learns to solve the problem with the minimal number of processing steps.
4. It is also called as DAVE-2 System by Nvidia

# Demo
<img src="https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif">

# Conclusions from the paper
1. This demonstrated that CNNs are able to learn the entire task of lane and road following without manual decomposition into road or lane marking detection, semantic abstraction, path planning, and control.The system learns for example to detect the outline of a road without the need of explicit labels during training.
2. A small amount of training data from less than a hundred hours of driving was sufficient to train the car to operate in diverse conditions, on highways, local and residential roads in sunny, cloudy, and rainy conditions.
3. The CNN is able to learn meaningful road features from a very sparse training signal (steering alone).
4. More work is needed to improve the robustness of the network, to find methods to verify the robust- ness, and to improve visualization of the network-internal processing steps.

Watch Real Car Running Autonoumously using this Algorithm https://www.youtube.com/watch?v=NJU9ULQUwng

A TensorFlow/keras implementation of this [Nvidia paper](https://arxiv.org/pdf/1604.07316.pdf) with some changes.

# How to Use
Download the [dataset](https://drive.google.com/file/d/0B-KJCaaF7elleG1RbzVPZWV4Tlk/view?usp=sharing) and extract into the repository folder

Use `python train.py` to train the model

Use `python run.py` to run the model on a live webcam feed

Use `python run_dataset.py` to run the model on the dataset

To visualize training using Tensorboard use `tensorboard --logdir=./logs`, then open http://0.0.0.0:6006/ into your web browser.

# Some other State of the Art Implementations
1. Implementations: https://github.com/udacity/self-driving-car
2. Blog: https://medium.com/udacity/teaching-a-machine-to-steer-a-car-d73217f2492c

# Credits & Inspired By
1. https://github.com/SullyChen/Autopilot-TensorFlow
2. Research paper: End to End Learning for Self-Driving Cars by Nvidia. [https://arxiv.org/pdf/1604.07316.pdf]
