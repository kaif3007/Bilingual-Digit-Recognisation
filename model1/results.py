from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report


test_datagen=ImageDataGenerator(rescale=1./255)



test_generator=test_datagen.flow_from_directory(
	'data1/test_eng_hin/hin',
	 target_size=(28,28),
	 color_mode='grayscale',
	 batch_size=32,
	 class_mode='categorical',
	 shuffle=False
	)

model=load_model('first_trail.h5')

pred=model.predict_generator(test_generator,steps=len(test_generator),verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)
labels=(test_generator.classes)


print(classification_report(labels,predicted_class_indices))