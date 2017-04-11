
## 1. Introduction

First of all let me say "Phew!". This was one of the most challenging projects I have worked on, since may be grad school! It was definitely challenging in terms of the amount of learning and implementation I had to do in such a small amount of time but it was worth it.

The aim of this project is to design a model(ConvNet) which learns from simulator images fed to it and is able to accurately nagivate a car, autonomously, around a track. The simulator(included in the repo) was designed and provided by Udacity.

## 2. The Data

The Dataset has a bunch of images captured from left, center and right cameras at a certain frame rate(from the simulator). Each image has an associated ID for which the steering angle, throttle, brake and speed is captured in a .csv file. The simulator(in autonomous mode) uses only the center camera's images as the feed for the model but as I will discuss below, I decided to use the images from all the cameras.

The dataset which eventually gave me the best results was the Udacity dataset. I really wish Udacity had emphasized that the dataset works really well for people without a joystick or controller handy. The model's performance on data captured by me using the simulator and the keyboard wasn't good enough.

Spending time on how the simulator works was worth it but not the time spent on capturing the right data. There was definitely confusing verbage in the slides of the project description because there was mentions about the user capturing the data, recovery data(this is was time consuming!) and data provided by Udacity. Eventually everything worked out for me, only after a lot of engineering but I definitely think the wording could have been less misleading.

## 3. ConvNet Model

After a bunch of reading and contemplating I decided to steer towards the NVIDIA model(pun intended). The NVIDIA model according the to paper had some great results: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

The model has the following layers(in order):

1. 3 Convolution layers with 2x2 strides and 5x5 kernels(with 24, 36 and 48 filters)
2. 2 Convolution layers with no(unit) strides and 3x3 kernels(with 64 filters each)
3. 3 Fully Connected(Dense) Layers with 100, 50, 10 outputs
4. Final output layer which predicts the steering angle

Here is a detailed visualization of the model:
<p align="center">
  <img src="https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png">
</p>

The model expects an input array in the form 66(height)x200(width)x3(channels). The images array was normalized and converted to YUV.

The paper doesn't go in depth about certain details like padding, activation or regularization techniques. It turned out(for me atleast), tweaking the model for overfitting was the biggest factor in making the model work in simulation mode.


## 4. Making it work

Selecting the model up and running was the least time consuming. The other parts like selecting the data, preprocessing the data, creating the generator, optimizing and regularization was the most time consuming parts of this project.

I definitely had to do step-by-step engineering to make the final model work for me. NVIDIA's model was hard to "tune" because the loss was so low from the get go because the model data was skewed towards driving straight.

Here is the distribution of the steering angle from Udacity's dataset:

<p align="center">
  <img src="https://s28.postimg.org/4bbqt2sil/Screen_Shot_2017_01_28_at_8_30_06_PM.png">
</p>
  
To counter this bias, one of the first things I did when preprocessing the dataset was to add an offset to the steering angles associated with the left(+ offset) and right(- offset) camera images. As pointed out by many in the forums, the idea behind adding/subtracting the offset is to, in effect, "bring the car back to the center". This is because the autonomous simulation only use images from the center camera. The other preprocessing was converting from RGB->YUV and normalization(same as the NVIDIA model, as mentioned earlier). I definitely used data augmentation using a generator because of memory constraints. One of the things I controlled during augmentation(this setting was a hyperparameter for my model) was the amount of data with zero steering angles allowed during each epoch of training.

With the above mentioned preprocessing I was getting a loss close to 0% even after 5 epochs, this definitely was suspicious because when I ran my model in autonomous mode, the car didn't even work for the first few autonomous mode yards because it would drive itself off the road. I knew at this point that my model was definitely overfitting.

So that lead me to my final(most time consuming) step, regularization to prevent overfitting. First, I changed the activation(RELU to ELU) of the output of all layers in the model except for the output layer. This change showed a little bit of improvement in loss decay but not much in terms of performance. The next step was to add/remove dropout from different layers and this didn't help me a whole lot, interestingly, because I thought dropout would surely help with overfitting. At this point I was quite exhausted and at the peak of frustration because I don't have a GPU and was running models(which took about 12-14 minutes) on EC2 and there was a lot of waiting around.

In the periods of waiting around, I read about L2 regularization and thought about using it on my ConvNet. ConvNets are all about extracting features of images and have a depth full of features which eventually help make predictions. My first instinct was to apply L2 regularization(weights and activity) on the 1st and 2nd dense layers but that pushed my loss way off and my model's loss was in the the range 13%-15%.

This was when the whole concept of ConvNets hit me("Must go deeper"), the deepest points of NVIDIA's model are the last 2 Convolution layers. Each of these had a depth of 64 at the output. So I decided to apply only weight regularization this time on the last 2 convolutional layers. The model's loss was definitely not too low but low enough(to begin with) and there was no steep drops in loss, after each epoch. This thought turned out to be great and my final model worked on Track1(and almost all of Track2 too, definitely can to be tweaked to make it work on Track2 as well)


## 5. Final Configuration

So, this is how my final configuration looks like:

1. Offset of +0.20 to steering angle of left camera
2. Offset of -0.25 to steering angle of right camera
3. Augmentation Threshold* = 0.5(decays at the rate of threshold/(1+epoch)
4. Batch Size = 128
6. Epochs = 8
7. Adam Optimizer(Learning Rate = 0.001)
8. L2 regularization on weights of last 2 Convolution layers
9. Dropout of 0.25 after Flattening

*This controls how much of the zero steering angle data is allowed during each epoch

## 6. Training, Validation and Driving(testing)

During training I did a 90%-10%-ish split between training and validation datasets. There was no test data, the testing was purely observing behavior of the model on the simulator.

Intuitively thinking I went for a lower throttle(0.14) for Track1 to prevent sharp turns(a human slows down at turns/corners) and a slightly higher throttle(0.22) for Track2 to get over some of the big hills.

## 7. Final Thoughts

I think I have made my feelings clear that this was a hard project :). Learnt a lot, made great progress in application of knowledge and playing an engineer to make this work. My model can be definitely tuned to make the drive smoother and also make it pass certain harder sections of Track 2(where it gets stuck at on one of the last big curves).

Steering angle(may be the biggest factor) is only one of the factors which makes a car driveable. The other things like throttle, brakes and speed also are big factors. So I am left with this one final thought "How to incorporate other factors to make the car drive efficiently? Since the model outputs only the steering angle..."
