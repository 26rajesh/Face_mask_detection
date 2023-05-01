#importing necessary packages
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
import random,shutil
from keras.models import Sequential
from keras.layers import Dropout,Conv2D,Flatten,Dense, MaxPooling2D, BatchNormalization
from keras.models import load_model

#defining a generator function
def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=True,batch_size=1,target_size=(24,24),class_mode='categorical' ):
    return gen.flow_from_directory(dir,batch_size=batch_size,shuffle=shuffle,color_mode='grayscale',class_mode=class_mode,target_size=target_size)

#Batch Size and Target Size
BS= 32
TS=(24,24)

#Loading the data
train_batch= generator('Face_mask/train',shuffle=True, batch_size=BS,target_size=TS)
valid_batch= generator('Face_mask/val',shuffle=True, batch_size=BS,target_size=TS)

#Steps per epoch and Validation steps
SPE= len(train_batch.classes)//BS
VS = len(valid_batch.classes)//BS

#Defining the Model Architecture
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24,24,1)),
    MaxPooling2D(pool_size=(1,1)),
    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D(pool_size=(1,1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1,1)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'), 
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

#Compiling the model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#Fitting the model
model.fit(train_batch,validation_data=valid_batch,epochs=15,steps_per_epoch=SPE,validation_steps=VS)

#Saving the model
model.save('model.h5')
