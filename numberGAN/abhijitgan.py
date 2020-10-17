#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf

## you may have to insert these lines if you have a GPU
#otherwise you can comment them out
tf.config.list_physical_devices('GPU')
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
##

import numpy as np
import matplotlib.pyplot as plt
import os


# In[3]:


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train.shape


# In[4]:


# Data has to be of type float32, reshaped and normalized
x_train = x_train.astype('float32')
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_train = (x_train-127.5)/127.5
x_train.shape


# In[5]:


x_train_data = tf.data.Dataset.from_tensor_slices(x_train).shuffle(x_train.shape[0]).batch(256)


# In[6]:


from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, Flatten, ReLU, BatchNormalization, Reshape, Dropout


# In[7]:


#generator model
#the generator takes some noise and creates an image of the same size as the train images
g_model = Sequential()
g_model.add(Dense(7*7*256, input_shape=(100,)))
g_model.add(Reshape((7,7,256)))
assert g_model.output_shape == (None, 7, 7, 256)

g_model.add(Conv2DTranspose(128, (5,5), strides=(1,1), padding='same', use_bias=False))
g_model.add(BatchNormalization())
g_model.add(ReLU())
assert g_model.output_shape == (None, 7, 7, 128)

g_model.add(Conv2DTranspose(64, (5,5), strides=(2,2), use_bias=False, padding='same'))
g_model.add(BatchNormalization())
g_model.add(ReLU())
assert g_model.output_shape == (None, 14, 14, 64)

g_model.add(Conv2DTranspose(1, (5,5), strides=(2,2), use_bias=False, padding='same', activation='tanh'))
assert g_model.output_shape == (None, 28, 28, 1)


# In[8]:


#discriminator model
#discriminator takes an image and decides whether it is real or fake (1 or 0)
d_model = Sequential()
d_model.add(Conv2D(64, (5,5), strides=(2,2), padding='same', activation='relu', input_shape=[28,28,1]))
d_model.add(Dropout(0.3))

d_model.add(Conv2D(128, (5,5), strides=(2,2), padding='same', activation='relu'))
d_model.add(Dropout(0.3))
d_model.add(Flatten())
d_model.add(Dense(1))


# In[9]:


## pass some noise to the generator and view generated image
noise = tf.random.normal([1,100])
g_img = g_model(noise)


# In[10]:


plt.imshow(g_img[0,:,:,0])


# In[11]:


print(d_model(g_img))


# In[12]:


# initializing the losses and optimizers for the 2 CNNs
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam


# In[13]:


bce = BinaryCrossentropy(from_logits=True)
g_optimizer = Adam(1e-4)
d_optimizer = Adam(1e-4)


# In[14]:


#generator loss
def g_loss(fake_img_output):
    return bce(tf.ones_like(fake_img_output), fake_img_output)


# In[15]:


#discriminator loss
def d_loss(real_img_output, fake_img_output):
    real_img_loss = bce(tf.ones_like(real_img_output), real_img_output)
    fake_img_loss = bce(tf.zeros_like(fake_img_output), fake_img_output)
    return real_img_loss + fake_img_loss


# In[16]:


from IPython import display

NUMEPOCHS = 100
seed = tf.random.normal([1, 100])

#train all batches for each epoch and output the image 
#from the generator for the above seed 

def train(x_train_data):
    for i in range(NUMEPOCHS):
        print(f"Epoch number {i}")
        for batch in x_train_data:
            train_batch(batch)
        
        display.clear_output(wait=True)
        display_image(g_model, seed)
        
    display.clear_output(wait=True)
    display_image(g_model, seed)
        


# In[17]:


@tf.function
#train each batch
#generate the images = batch size from the generator (fake images)
#get the outputs from the discriminator for the real and fake images
#calculate the losses using binary cross entropy
#calculate and apply the gradients

def train_batch(img_batch):
    noise = tf.random.normal([256, 100])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_images = g_model(noise, training=True)
        
        real_img_output = d_model(img_batch, training=True)
        fake_img_output = d_model(gen_images, training=True)
        
        generator_loss = g_loss(fake_img_output)
        discriminator_loss = d_loss(real_img_output, fake_img_output)
        
        g_gradient = gen_tape.gradient(generator_loss, g_model.trainable_variables)
        d_gradient = disc_tape.gradient(discriminator_loss, d_model.trainable_variables)
        
        g_optimizer.apply_gradients(zip(g_gradient, g_model.trainable_variables))
        d_optimizer.apply_gradients(zip(d_gradient, d_model.trainable_variables))


# In[18]:


def display_image(gen, seed):
    pred = gen(seed, training=False)
    plt.imshow(pred[0,:,:,0])
    plt.show()


# In[19]:


train(x_train_data)


# In[27]:


## use the generator to create another digit using new noise data
newnoise = tf.random.normal([1,100])
new_img = g_model(newnoise)
plt.imshow(new_img[0,:,:,0])
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




