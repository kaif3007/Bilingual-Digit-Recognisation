from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation='relu'))


model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

print(model.summary())

model.compile(optimizer=optimizers.RMSprop(lr=1e-4),loss='categorical_crossentropy',metrics=['accuracy'])

train_datagen=ImageDataGenerator(rescale=1./255,
	 rotation_range=50,
	 width_shift_range=0.2,
	 height_shift_range=0.2,
	 zoom_range=0.2
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
	steps_per_epoch=2000,epochs=40,
	validation_data=valid_generator,validation_steps=300)

model.save('first_trail.h5')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()