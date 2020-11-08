from keras.layers import Conv2D,Lambda,MaxPooling2D
from keras.layers import Dense,Dropout,Flatten
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import pickle



model=Sequential()
model.add(Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(BatchNormalization())

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(BatchNormalization())

model.add(Conv2D(256,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))


model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(512,activation='relu'))
model.add(Dense(10,activation='softmax'))

print(model.summary())

model.compile(optimizer=optimizers.Adam(),loss='categorical_crossentropy',metrics=['accuracy'])

train_datagen=ImageDataGenerator(rescale=1./255,
	 rotation_range=10,
	 width_shift_range=0.1,
	 height_shift_range=0.1,
	 zoom_range=0.1
	 )

valid_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(
	'data1/train',
	 target_size=(28,28),
	 color_mode='grayscale',
	 batch_size=32,
	 class_mode='categorical',
	 shuffle=True
	)


valid_generator=valid_datagen.flow_from_directory(
	'data1/validation',
	 target_size=(28,28),
	 color_mode='grayscale',
	 batch_size=32,
	 class_mode='categorical',
	 shuffle=True
	)

'''
for data,label in train_generator:
	print(data.shape)
	print(label.shape)
	break
'''

history=model.fit_generator(train_generator,
	steps_per_epoch=2000,epochs=50,
	validation_data=valid_generator,validation_steps=300)

model.save('second_trail.h5')

with open('trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)