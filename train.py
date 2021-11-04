# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 17:07:20 2021

@author: s1253
"""

import os
import time
import pathlib
import tensorflow as tf
from ops import preprocess_image, generate_and_save_images
from Model import Generator, Discriminator
from tensorflow.keras import backend as K


TARGET_IMG_SIZE = 64 

BATCH_SIZE = 28
NOISE_DIM = 100
LAMBDA = 10 

EPOCHs = 40
CURRENT_EPOCH = 1 
SAVE_EVERY_N_EPOCH = 5

N_CRITIC = 3 
LR = 1e-4
MIN_LR = 0.000001 
DECAY_FACTOR=1.00004 


data_path = pathlib.Path('faces_less')
file_list = [str(path) for path in data_path.glob('*.jpg')]

list_ds = tf.data.Dataset.from_tensor_slices(file_list)

#data preprocess
train_data = list_ds.map(preprocess_image).shuffle(500).batch(BATCH_SIZE)



MODEL_NAME = 'WGAN'
OUTPUT_PATH = os.path.join('outputs', "LayerNorm")


if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)



G_optimizer = tf.keras.optimizers.Adam(0.0001,beta_1=0.5)
D_optimizer = tf.keras.optimizers.Adam(0.0001,beta_1=0.5)


generator = Generator(input_dim=NOISE_DIM,
                      generator_shape=(4, 4, 1024),
                      batch_norm=True,
                      activation="leaky_relu",
                      dropout=0.2,
                      generator_upsample=[2,2,2,2],
                      generator_conv_filters=[512,256,128,3],
                      generator_conv_kernal=[3,3,3,3],
                      generator_conv_stride=[1,1,1,1]
                      ).build_layer()


discriminator = Discriminator(discriminator_input=(TARGET_IMG_SIZE, TARGET_IMG_SIZE, 3),
                         layer_norm=True,
                         activation="leaky_relu",
                         discriminator_conv_filters=[64,128,256,512,1],
                         discriminator_conv_kernal=[3,3,3,3,3],
                         discriminator_conv_stride=[1,2,2,2,2]
                         ).build_discriminator()
    
#generator.summary()
#discriminator.summary()


checkpoint_path = os.path.join("checkpoints", "picture", MODEL_NAME)

ckpt = tf.train.Checkpoint(generator=generator,
                           discriminator=discriminator,
                           G_optimizer=G_optimizer,
                           D_optimizer=D_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    latest_epoch = int(ckpt_manager.latest_checkpoint.split('-')[1])
    
    CURRENT_EPOCH = latest_epoch * SAVE_EVERY_N_EPOCH
    print ('Latest checkpoint of epoch {} restored!!'.format(CURRENT_EPOCH))
    
 
    
''' training step  '''

def WGAN_GP_train_d_step(real_image, batch_size):

    noise = tf.random.normal([batch_size, NOISE_DIM])
    epsilon = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0, maxval=1)

    with tf.GradientTape(persistent=True) as d_tape:
        with tf.GradientTape() as gp_tape:
            fake_image = generator([noise], training=True)
            fake_image_mixed = epsilon * tf.dtypes.cast(real_image, tf.float32) + ((1 - epsilon) * fake_image)
            fake_mixed_pred = discriminator([fake_image_mixed], training=True)
            
        grads = gp_tape.gradient(fake_mixed_pred, fake_image_mixed)
        grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean(tf.square(grad_norms - 1))
        
        fake_pred = discriminator([fake_image], training=True)
        real_pred = discriminator([real_image], training=True)
        
        D_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred) + LAMBDA * gradient_penalty
    
    D_gradients = d_tape.gradient(D_loss,
                                            discriminator.trainable_variables)
    
    D_optimizer.apply_gradients(zip(D_gradients,
                                                discriminator.trainable_variables))




def WGAN_GP_train_g_step(real_image, batch_size):
   
    noise = tf.random.normal([batch_size, NOISE_DIM])

    with tf.GradientTape() as g_tape:
        fake_image = generator([noise], training=True)
        fake_pred = discriminator([fake_image], training=True)
        G_loss = -tf.reduce_mean(fake_pred)
        
    G_gradients = g_tape.gradient(G_loss,
                                            generator.trainable_variables)
   
    G_optimizer.apply_gradients(zip(G_gradients,
                                                generator.trainable_variables))


current_learning_rate = LR
n_critic_count = 0

# customize learning rate
def learning_rate_decay(current_lr, decay_factor=DECAY_FACTOR):
    
    new_lr = max(current_lr / decay_factor, MIN_LR)
    return new_lr

def set_learning_rate(new_lr):
    K.set_value(D_optimizer.lr, new_lr)
    K.set_value(G_optimizer.lr, new_lr)


num_examples_to_generate = 18
sample_noise = tf.random.normal([num_examples_to_generate, NOISE_DIM])     



for epoch in range(CURRENT_EPOCH, EPOCHs+1):
    start = time.time()
    print('Start of epoch %d' % (epoch,))

    current_learning_rate = learning_rate_decay(current_learning_rate)
    print('current_learning_rate %f' % (current_learning_rate,))
    set_learning_rate(current_learning_rate)
    
    for step, (image) in enumerate(train_data):
        current_batch_size = image.shape[0]
        # Train critic (discriminator)
        WGAN_GP_train_d_step(image, batch_size=current_batch_size)
        n_critic_count += 1
        if n_critic_count >= N_CRITIC: 
            # Train generator
            WGAN_GP_train_g_step(image, batch_size=current_batch_size)
            n_critic_count = 0
        
        if step % 10 == 0:
            print ('.', end='')
    
    generate_and_save_images(generator, epoch, [sample_noise], OUTPUT_PATH, figure_size=(12,6), subplot=(3,6), save=True)
    
    if epoch % SAVE_EVERY_N_EPOCH == 0:
        #save model
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch,
                                                             ckpt_save_path))
    
    print ('Time taken for epoch {} is {} sec\n'.format(epoch,
                                                      time.time()-start))



















