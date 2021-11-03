# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 12:31:49 2021

@author: s1253
"""

import tensorflow as tf
import os
import matplotlib.pyplot as plt

TARGET_IMG_SIZE = 64 

def normalize(image):
    '''
        normalizing the images to [-1, 1]
    '''
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = (image - 127.5) / 127.5
    return image

def preprocess_image(file_path):
    images = tf.io.read_file(file_path)
    # convert the compressed string to a 3D uint8 tensor
    images = tf.image.decode_jpeg(images, channels=3)
    images = tf.image.resize(images, (TARGET_IMG_SIZE, TARGET_IMG_SIZE))
                          
    images = normalize(images)
    return images


    
    
def generate_and_save_images(model, epoch, test_input, OUTPUT_PATH,figure_size=(12,6), subplot=(3,6), save=True):
    predictions = model.predict(test_input)
    
    for i in range(predictions.shape[0]):
        axs = plt.subplot(subplot[0], subplot[1], i+1)
        axs.imshow(predictions[i] * 0.5 + 0.5)
        plt.axis('off')
    if save:
        plt.savefig(os.path.join(OUTPUT_PATH, 'image_at_epoch_{:04d}.png'.format(epoch)))
    plt.show()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    