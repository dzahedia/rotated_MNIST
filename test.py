from __future__ import print_function
from keras.datasets import mnist
from keras.models import  load_model


from datetime import datetime
import numpy as np

import cv2
borderType = cv2.BORDER_CONSTANT
start = datetime.now()


# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


saved_model = 'rotated_21_epoch.h5'
err = []
model = load_model(saved_model)
p = model.predict(x_test.reshape(10000,28,28,1))
yhat = np.argmax(p,axis = 1)
for i in range(10000):
    if yhat[i] != y_test[i]:
        err.append(i)
print('accuracy', 1- (len(err)/10000))