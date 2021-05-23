#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install split-folders


# In[2]:


from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Concatenate, Dot, Lambda, Input, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D 
from keras.optimizers import Adam
import os
from keras.preprocessing import image
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
import splitfolders
from keras import backend as K
from keras.callbacks import EarlyStopping, TensorBoard 

splitfolders.fixed('/content/drive/MyDrive/Project/Segmented Medicinal Leaf Images',output='/content/drive/MyDrive/Project/Dataset', seed=123, fixed=(5,10))
# In[3]:


def get_data(datadir):
  categories = os.listdir(datadir)
  data = []
  for category in categories:
    path = os.path.join(datadir, category)
    labels = categories.index(category)
    for imgpath in os.listdir(path):
      imgs = image.load_img(os.path.join(path, imgpath), target_size=(224,224))
      data.append((imgs, labels))
  
  random.shuffle(data)

  X_img = []
  y_lab = []

  for img , label in data:
    X_img.append(img)
    y_lab.append(label)

  X_list = [ image.img_to_array(img) for img in X_img]
  X = np.asarray(X_list)

  X /= 255

  y = np.asarray(y_lab)

  return X,y


train_data_dir = 'few shot/test'
Xtrain, ytrain = get_data(train_data_dir)

# test_data_dir = '/content/drive/MyDrive/Project/Dataset/test'
# Xtest, ytest = get_data(test_data_dir)

val_data_dir =  'few shot/val'
Xval , yval = get_data(val_data_dir)


# In[4]:


Xtrain.shape, ytrain.shape

plt.figure(figsize=(5,5))
plt.imshow(Xtrain[17])
plt.axis('off')
plt.show()
# In[5]:


def make_pairs(x,y):
  num_classes = max(y) + 1
  digit_indices = [np.where(y==i)[0] for i in range(num_classes)]

  pairs = []
  labels = []
  
  for idx1 in range(len(x)):
    x1 = x[idx1]
    label1 = y[idx1]
    idx2 = random.choice(digit_indices[label1])
    x2 = x[idx2]

    pairs += [[x1, x2]] 
    labels += [1]

    label2 = random.randint(0,num_classes-1)
    while label2 == label1:
      label2 = random.randint(0,num_classes-1)

    idx2 = random.choice(digit_indices[label2])
    x2 = x[idx2]

    pairs += [[x1, x2]]
    labels += [0]

  return np.array(pairs), np.array(labels)


pairs_train , labels_train = make_pairs(Xtrain, ytrain)

# pairs_test , labels_test = make_pairs(Xtest, ytest)

pairs_val , labels_val = make_pairs(Xval, yval)


# In[ ]:





# In[6]:


pairs_train.shape


# In[7]:


labels_train.shape


# In[8]:


plt.imshow(pairs_train[3,0])
plt.axis('off')
print(labels_train[3])

def euclidean_distance(vects):
  x, y = vects
  sum_square = K.sum(K.square(x - y), axis = 1 , keepdims=True)
  return K.sqrt(K.maximum(sum_square, K.epsilon()))

input = Input((299,299,3))
x = Conv2D(filters = 128, kernel_size = 3, padding="same", activation="relu")(input)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.3)(x)

x = Conv2D(filters = 128, kernel_size = 3, padding="same", activation="relu")(x)
x = MaxPooling2D(pool_size=2)(x)
x = Dropout(0.3)(x)

x = Conv2D(filters = 64, kernel_size = 3, padding="same", activation="relu")(x)
x = MaxPooling2D(pool_size=2)(x)
x = Dropout(0.3)(x)
 
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
model = Model(input, x)

input1 = Input((299,299,3))
input2 = Input((299,299,3))

sis1 = model(input1)
sis2 = model(input2)

merge_layer = Lambda(euclidean_distance)([sis1, sis2])
dense_layer = Dense(1, activation='sigmoid')(merge_layer)
siamese_model = Model(inputs=[input1, input2], outputs=dense_layer)
this cell is not running leave it

# In[9]:


def euclidean_distance(vects):
  x, y = vects
  sum_square = K.sum(K.square(x - y), axis = 1 , keepdims=True)
  return K.sqrt(K.maximum(sum_square, K.epsilon()))


# In[10]:


# import necessary library
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import Sequential
from keras.layers import *


# In[11]:


# load and creare a model object
v2model = MobileNetV2(input_shape=[224,224,3], weights='imagenet', include_top=False)
# model summary
v2model.summary()


# In[12]:


for layer in v2model.layers:
    layer.trainable = False


# In[13]:


x = Flatten()(v2model.output)


# In[14]:


prediction = Dense(30, activation='softmax')(x)


# In[15]:


from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model


# In[16]:


# create a model object
modelv2 = Model(inputs=v2model.input, outputs=prediction)


# In[17]:


modelv2.summary()


# In[18]:


## 2nd model
import tensorflow as tf


# In[19]:


model2 = tf.keras.applications.NASNetMobile(input_shape=(224,224,3), weights='imagenet', include_top=False)


# In[20]:


for layer in model2.layers:
    layer.trainable = False


# In[21]:


y = Flatten()(model2.output)
prediction2 = Dense(30, activation='softmax')(y)


# In[22]:


model2 = Model(inputs=model2.input, outputs=prediction2)


# In[23]:


model2.summary()


# In[24]:


input1 = Input((224,224,3))
input2 = Input((224,224,3))


# In[25]:


sis1 = modelv2(input1)
sis2 = model2(input2)


# In[26]:


merge_layer = Lambda(euclidean_distance)([sis1, sis2])
dense_layer = Dense(30, activation='softmax')(merge_layer)
siamese_model = Model(inputs=[input1,input2], outputs=dense_layer)

input1 = Input((224,224,3))
input2 = Input((224,224,3))
# In[27]:


siamese_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
siamese_model.summary()


# In[28]:


callbacks = EarlyStopping(monitor='val_accuracy', patience=3,verbose=1)


# In[32]:


siamese_model.fit([pairs_train[:,0], pairs_train[:,1]], labels_train[:], batch_size=32, epochs=30, validation_data=([pairs_val[:,0], pairs_val[:,1]], labels_val[:]), callbacks=callbacks)


# In[30]:


# siamese_model.evaluate([pairs_test[:,0], pairs_test[:,1]], labels_test[:])


# In[31]:


#now try to run

