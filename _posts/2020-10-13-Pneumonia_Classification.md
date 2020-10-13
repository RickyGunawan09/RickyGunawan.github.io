---
title: "Pneumonia Classification"
date: 2020-10-13
tags: [Pneumonia,X-ray, data science, machine learning, Computer Vision]
header:
  image: "/images/xray.jpg"
excerpt: "Pneumonia,X-ray, data science, machine learning, Computer Vision"
mathjax: "true"
---

**Pneumonia**
an acute disease that is marked by inflammation of lung tissue accompanied by infiltration of alveoli and often bronchioles with white blood cells (such as neutrophils) and fibrinous exudate, is characterized by fever, chills, cough, difficulty in breathing, fatigue, chest pain, and reduced lung expansion, and is typically caused by an infectious agent (such as a bacterium, virus, or fungus). The disease may be classified by where it was acquired, such as community- or hospital-acquired or healthcare-associated pneumonia.

<img src="{{ site.url }}{{ site.baseurl }}/images/output_image/pneumonia.jpg" alt="linearly separable data">

```python
zip_path = '/content/drive/My\ Drive/BCML/FinalProject/chest_xray_backup.zip'

!cp {zip_path} /content/

!cd /content/

!unzip -q /content/chest_xray_backup.zip -d /content

!rm /content/chest_xray_backup.zip
```


```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import cv2
import datetime
import pandas as pd
from numpy import expand_dims

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc

import keras
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization,MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD , RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from keras.models import load_model

```

# **Definisi Function**


```python
labels = [ 'NORMAL','PNEUMONIA']
img_size = 200
dataset_dir_train = '/content/chest_xray_backup/train'
dataset_dir_val = '/content/chest_xray_backup/val'
dataset_dir_test = '/content/chest_xray_backup/test'

def get_training_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) 
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)

def process_data(img):
    img = np.array(img)/255.0
    img = np.reshape(img, (200,200,1))
    return img

def compose_dataset(df):
    data = []
    labels = []

    for img, label in df.values:
        data.append(process_data(img))
        labels.append(label)
        
    return np.array(data), np.array(labels)

def imagedatagenerator(param):
    test_Image_DG=cv2.imread('/content/chest_xray_backup/train/NORMAL/IM-0115-0001.jpeg', cv2.IMREAD_GRAYSCALE)   # Reading an Image
    test_Image_DG=cv2.resize(test_Image_DG,(200,200))              
    test_Image_DG = np.array(test_Image_DG)/255.0
    test_Image_DG = np.reshape(test_Image_DG, (200,200,1))
    input_image_DG= np.expand_dims(test_Image_DG, axis=0)

    if param == "rotation_range":
        datagen_rotation_range = ImageDataGenerator(rotation_range =30)
        it = datagen_rotation_range.flow(input_image_DG, batch_size=1)
    elif param == "zoom_range":
        datagen_zoom_range = ImageDataGenerator(zoom_range =0.2)
        it = datagen_zoom_range.flow(input_image_DG, batch_size=1)
    elif param == "width_shift_range":
        datagen_width_shift = ImageDataGenerator(width_shift_range =0.1)
        it = datagen_width_shift.flow(input_image_DG, batch_size=1)
    elif param == "height_shift_range":
        datagen_height_shift_range = ImageDataGenerator(height_shift_range =0.1)
        it = datagen_height_shift_range.flow(input_image_DG, batch_size=1)
    elif param == "horizontal_flip":
        datagen_horizontal_flip = ImageDataGenerator(horizontal_flip =True)
        it = datagen_horizontal_flip.flow(input_image_DG, batch_size=1)
    elif param == "All":
        datagen_all = ImageDataGenerator(rotation_range =30,zoom_range =0.2,width_shift_range =0.1,height_shift_range =0.1,horizontal_flip =True)
        it = datagen_all.flow(input_image_DG, batch_size=1)

    fig=plt.figure(figsize=(30,30))
    for i in range(8):
        ax=fig.add_subplot(1,8,i+1)
        batch = it.next()
        image = batch[0]
        image = image[:,:,0]
        ax.imshow(image,cmap='gray')
    return
```

# **Mengambil Data Image**


```python
train = get_training_data(dataset_dir_train)
test = get_training_data(dataset_dir_test)
val = get_training_data(dataset_dir_val)
```


```python
train[0][0]
```


```python
labels[train[0][1]]
```


```python
labels[train[-1][1]]
```


```python
train_df = pd.DataFrame(train, columns=['image', 'label'])
test_df = pd.DataFrame(test, columns=['image', 'label'])
val_df = pd.DataFrame(val, columns=['image', 'label'])
```


```python
print(train_df)
```

# **Visualisasi Data**


```python
plt.figure(figsize=(20,5))

plt.subplot(1,3,1)
sns.countplot(train_df['label'])
plt.title('Train data')

plt.subplot(1,3,2)
sns.countplot(test_df['label'])
plt.title('Test data')

plt.subplot(1,3,3)
sns.countplot(val_df['label'])
plt.title('Validation data')

plt.show()
```


```python
plt.figure(figsize = (5,5))
plt.imshow(train[0][0], cmap='gray')
plt.title(labels[train[0][1]])

plt.figure(figsize = (5,5))
plt.imshow(train[-1][0], cmap='gray')
plt.title(labels[train[-1][1]])
```

# **Feature Engineering**


```python
feature_train, label_train = compose_dataset(train_df)
feature_test, label_test = compose_dataset(test_df)
feature_val, label_val = compose_dataset(val_df)
```


```python
print('Train data shape: {}, Labels shape: {}'.format(feature_train.shape, label_train.shape))
print('Test data shape: {}, Labels shape: {}'.format(feature_test.shape, label_test.shape))
print('Validation data shape: {}, Labels shape: {}'.format(feature_val.shape, label_val.shape))
```


```python
print(label_val)
```


```python
train_datagen = ImageDataGenerator(rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
                                   zoom_range = 0.2, # Randomly zoom image 
                                   width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                                   height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                                   horizontal_flip = True)  # randomly flip images

train_datagen.fit(feature_train)
```


```python
label_train = to_categorical(label_train)
label_test = to_categorical(label_test)
label_val = to_categorical(label_val)
```

## **Visualisasi Image Data Generator yang di gunakan**


```python
list_param=["rotation_range","zoom_range","width_shift_range","height_shift_range","horizontal_flip","All"]

for i in list_param:
    imagedatagenerator(i)
```

# **Model CNN**


```python
model = Sequential()

model.add(Conv2D(filters=8, kernel_size=(7,7), padding='same', activation='relu', input_shape=(200, 200, 1)))
model.add(Conv2D(filters=8, kernel_size=(7,7), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(filters=16, kernel_size=(5,5), padding='same', activation='relu'))
model.add(Conv2D(filters=16, kernel_size=(5,5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

optimizer = Adam(lr=0.0001, decay=1e-5)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()
```


```python
i=1
for layer in model.layers:
    if 'conv' in layer.name: 
        filters, bias= layer.get_weights()
        print('Filters Shape: '+ str(filters.shape, )+" " + 'Bias Shape: '+str(bias.shape)+ "<---- layer: "+str(i))
        print("-----------")
        i=i+1
```

## **visualisasi Layer Filter CNN ke 1**


```python
layer= model.layers
layer_1= layer[0]
filter_1, bias_1= layer_1.get_weights()
print(filter_1.shape, bias_1.shape)

#Normalize the weights
f_min, f_max = filter_1.min(), filter_1.max()
filter_1 = (filter_1 - f_min) / (f_max - f_min)
filter_1 = filter_1[:,:,0]

fig= plt.figure(figsize=(20,20))
for i in range(8):
    ax = fig.add_subplot(1,8,i+1)
    ax.imshow(filter_1[:,:,i], cmap='gray')
```

## **visualisasi Layer Filter CNN ke 2**


```python
layer_2= layer[1]
filter_2, bias_2= layer_2.get_weights()
print(filter_2.shape, bias_2.shape)

#Normalize the weights
f_min2, f_max2 = filter_2.min(), filter_2.max()
filter_2 = (filter_2 - f_min2) / (f_max2 - f_min2)
filter_2 = filter_2[:,:,0]

fig= plt.figure(figsize=(20,20))
for i in range(8):
    ax = fig.add_subplot(1,8,i+1)
    ax.imshow(filter_2[:,:,i], cmap='gray')
```

## **Hasil Implementasi filter CNN ke 1**


```python
inp= model.inputs 
out1= model.layers[0].output  
feature_map_1= Model(inputs= inp, outputs= out1)  

img=cv2.imread('/content/chest_xray_backup/train/NORMAL/IM-0115-0001.jpeg', cv2.IMREAD_GRAYSCALE)   # Reading an Image
img=cv2.resize(img,(200,200))               # Resizing an Image
img = np.array(img)/255.0
img = np.reshape(img, (200,200,1))
input_img= np.expand_dims(img, axis=0)      # Expanding the dimension
print(input_img.shape)                      # Printing out the size of the Input Image
#-------------------------------------#---------------------------
f1=feature_map_1.predict(input_img)        # predicting out the Image 
print(f1.shape)                            # Let's see the shape
fig= plt.figure(figsize=(20,20))
for i in range(8):
    ax=fig.add_subplot(1,8,i+1)
    ax.imshow(f1[0,:,:,i],cmap='gray')
```

## **Hasil Implementasi filter CNN ke 2**


```python
inp= model.inputs 
out2= model.layers[1].output  
feature_map_2= Model(inputs= inp, outputs= out2)  

f2=feature_map_2.predict(input_img)        # predicting out the Image 
print(f2.shape)                            # Let's see the shape
fig= plt.figure(figsize=(20,20))
for i in range(8):
    ax=fig.add_subplot(1,8,i+1)
    ax.imshow(f2[0,:,:,i],cmap='gray')
```

# **Melatih Model pada Validation Set**


```python
filepath="weights_best.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) # Tempat dimana log tensorboard akan di
callbacks_list.append(TensorBoard(logdir, histogram_freq=1))
```


```python
history = model.fit(train_datagen.flow(feature_train,label_train, batch_size=4), validation_data=(feature_val, label_val), epochs = 25, verbose = 1, callbacks=callbacks_list)
```


```python
%load_ext tensorboard

%tensorboard --logdir logs
```

# **Evaluasi Model Pada data test**


```python
print("Loss of the model is - " , model.evaluate(feature_test,label_test)[0])
print("Accuracy of the model is - " , model.evaluate(feature_test,label_test)[1]*100 , "%")
```


```python
predictions = np.argmax(model.predict(feature_test), axis=-1)
predictions = predictions.reshape(1,-1)[0]
predictions[:100]
```


```python
label_test_hat = model.predict(feature_test, batch_size=4)
label_test_hat = np.argmax(label_test_hat, axis=1)
label_test = np.argmax(label_test, axis=1)
```


```python
# calculate confusion matrix & classification report
conf_m = confusion_matrix(label_test, label_test_hat)
clas_r = classification_report(label_test, label_test_hat,target_names = ['Normal (Class 0)','Pneumonia (Class 1)'])

# plot confusion matrix as heatmap
plt.figure(figsize=(5,3))
sns.set(font_scale=1.2)
ax = sns.heatmap(conf_m, annot=True,xticklabels=['H', 'P'], yticklabels=['H', 'P'], cbar=False, cmap='Blues',linewidths=1, linecolor='black', fmt='.0f')
plt.yticks(rotation=0)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
ax.xaxis.set_ticks_position('top') 
plt.title('Confusion matrix - test data\n(H - healthy/normal, P - pneumonia)')
plt.show()

# print classification report
print('Classification report on test data')
print(clas_r)
```

# **Visualisasi Data Validation**


```python
label_val_hat = model.predict(feature_val, batch_size=4)
label_val_hat = np.argmax(label_val_hat, axis=1)
label_val = np.argmax(label_val, axis=1)
```


```python
plt.figure(figsize=(20,15))
for i,x in enumerate(feature_val):
    plt.subplot(4,4,i+1)
    plt.imshow(x.reshape(200, 200), cmap='gray')
    plt.axis('off')
    plt.title('Predicted: {}, Real: {}'.format(label_val_hat[i], label_val[i]))
```

# **RUC-AUC Curve**

## **Data Test**


```python
label_pred = model.predict(feature_test,batch_size= 4)
label_pred = np.argmax(label_pred, axis=1)
```


```python
fpr, tpr, threshold = roc_curve(label_test, label_pred)
auc_test = auc(fpr, tpr)
```


```python
plt.plot(fpr, tpr, marker='.', label='cnn (area = {:.3f})'.format(auc_test))
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()
```

## **Data Train**


```python
train_pred = model.predict(feature_train,batch_size= 4)
train_pred = np.argmax(train_pred, axis=1)
train_label_roc = np.argmax(label_train, axis=1)
```


```python
fpr_train, tpr_train, threshold = roc_curve(train_label_roc, train_pred)
auc_train = auc(fpr_train, tpr_train)
```


```python
plt.plot(fpr_train, tpr_train, marker='.', label='cnn (area = {:.3f})'.format(auc_train))
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()
```