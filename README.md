# WGAN-GP
<p align="center"><img width="300px" src="https://github.com/Yukino1010/WGAN-GP/blob/master/picture/1.png" /><img width="300px" src="https://github.com/Yukino1010/WGAN-GP/blob/master/picture/2.png" /></p>
<br>
<p align="center"><img width="300px" src="https://github.com/Yukino1010/WGAN-GP/blob/master/picture/3.png" /><img width="300px" src="https://github.com/Yukino1010/WGAN-GP/blob/master/picture/4.png" /></p>
## Introduce
Wgan-gp is one of the most important changes on the generative models. Compare with wgan it just add a L2 penalty behind the loss function and replace Batch Normalization with Layer Normalization. What's amazing is that it can generate more stable and higher-quality pictures. 

## Network Structure
**generator architecture**
![image](https://github.com/Yukino1010/WGAN-GP/blob/master/picture/Generator.png)

### network design
- Replace transposed convolutions with  upsampling2d and conv2d
- Remove Batch Normalization in discriminator or use Layer Normalization 
- add gradient penalty behind loss function
- use Leaky-Relu behind all the CNN

## Hyperparameters
- batch_size = 28
- gradient-penalty lambda = 10
- learning rate : learning rate decay
- dropout = 0.2

<br>
because the limitation of the computer performance, I only can use batch-size of 28 and ten thousand of pictures(the original data have 70000 pictures)
I think the result will be better if I use bigger batch-size and more data
<br>

## Data
This implementation uses the data from Hung-yi Lee the professor in National Taiwan University


