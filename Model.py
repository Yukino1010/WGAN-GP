# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 15:03:37 2021

@author: s1253
"""
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Activation, BatchNormalization,\
    LeakyReLU, Dropout, UpSampling2D, Conv2DTranspose, LayerNormalization
import numpy as np

from tensorflow.keras import Model


    
class Generator():
    def __init__(self,
                 input_dim,
                 generator_shape,
                 batch_norm,
                 activation,
                 dropout,
                 generator_upsample,
                 generator_conv_filters,
                 generator_conv_kernal,
                 generator_conv_stride,
                 ):
        
        self.input_dim = input_dim
        self.n_layer = len(generator_conv_filters)
        self.generator_shape = generator_shape
        self.batch_norm = batch_norm
        self.activation = activation
        self.dropout = dropout
        self.generator_upsample = generator_upsample
        self.generator_conv_filters = generator_conv_filters
        self.generator_conv_kernal = generator_conv_kernal
        self.generator_conv_stride = generator_conv_stride
    
    def get_activation(self, activation):
    
        if activation == 'leaky_relu':
            layer = LeakyReLU(alpha = 0.2)
        else:
            layer = Activation(activation)
            
        return layer
    
    def build_layer(self):
        
        generator_input = Input(self.input_dim, name="generator_input")
        
        x = generator_input
        
        x = Dense(np.prod(self.generator_shape))(x)
        
        if self.batch_norm:
            x = BatchNormalization()(x)
        
        x = self.get_activation(self.activation)(x)
        
        x = Reshape(self.generator_shape)(x)
        
        if self.dropout:
            x = Dropout(self.dropout)(x)
          
        for i in range(self.n_layer):
            if self.generator_upsample[i] == 2:
                x = UpSampling2D()(x)
                x = Conv2D(
                    filters=self.generator_conv_filters[i],
                    kernel_size=self.generator_conv_kernal[i],
                    strides=self.generator_conv_stride[i],
                    padding="same",
                    name="generator_conv_"+str(i)
                    )(x)
            else:
                x = Conv2DTranspose(
                    filters = self.generator_conv_filters[i]
                    , kernel_size = self.generator_conv_kernal[i]
                    , padding = 'same'
                    , strides = self.generator_conv_stride[i]
                    , name = 'generator_conv_' + str(i)
                    )(x)
                    
            if i < self.n_layer-1:
                
                if self.batch_norm:
                    x = BatchNormalization()(x)
                    
                x = self.get_activation(self.activation)(x)
                
            else:
                x = Activation('tanh', name="tanh")(x)
                
        
        generator_out = x
        generator = Model(generator_input, generator_out)
        
        return generator
    
    
        
class Discriminator():
    def __init__(self,
                 discriminator_input,
                 layer_norm,
                 activation,
                 discriminator_conv_filters,
                 discriminator_conv_kernal,
                 discriminator_conv_stride,
                 ):
        
        self.discriminator_input = discriminator_input
        self.n_layer = len(discriminator_conv_filters)
        self.layer_norm = layer_norm
        self.activation = activation
        self.discriminator_conv_filters = discriminator_conv_filters
        self.discriminator_conv_kernal = discriminator_conv_kernal
        self.discriminator_conv_stride = discriminator_conv_stride
        
    def get_activation(self, activation):
    
        if activation == 'leaky_relu':
            layer = LeakyReLU(alpha = 0.2)
        else:
            layer = Activation(activation)
            
        return layer
    
    def build_discriminator(self):
        discriminator_input = Input(self.discriminator_input, name="dis_input")
        
        x =discriminator_input
        
        for i in range(self.n_layer):
             x = Conv2D(
                filters=self.discriminator_conv_filters[i],
                kernel_size=self.discriminator_conv_kernal[i],
                strides=self.discriminator_conv_stride[i],
                padding="same",
                name="discriminator_conv_"+str(i)
                )(x)
             
             if i < self.n_layer-1:
                 
                 if self.layer_norm:
                    x = LayerNormalization()(x)
                    
                 x = self.get_activation(self.activation)(x)
            
             else:
                 x = Flatten()(x)
                 x = Dense(1)(x)
                
        discriminate_out = x
        
        discriminator = Model(discriminator_input, discriminate_out)
        
        return discriminator
                         
        
    
       
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        