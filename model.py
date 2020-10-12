# import library
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import keras 
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, Flatten, Dropout


# load train data
df_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
Y = df_train['label']
Y_input = to_categorical(Y, num_classes = 10)
data_train = df_train.drop(['label'], axis = 1)

# load test data
df_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')


# scale the data
s = MinMaxScaler()
s.fit(data_train)
data_scaled = s.transform(data_train)
data_input = data_scaled.reshape(42000, 28, 28, 1)
test_data = s.transform(df_test)
test_set = test_data.reshape(-1, 28, 28, 1)


# create model
def create_model():
    model = Sequential()
    model.add(Conv2D(32, input_shape = (28, 28, 1), kernel_size = (3,3), strides = (1, 1), padding ='same'))
    model.add(LeakyReLU(0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding = 'same', strides =(1, 1)))
    model.add(LeakyReLU(0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), padding = 'same', strides =(1, 1)))
    model.add(LeakyReLU(0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(250))
    model.add(LeakyReLU(0.1))
    model.add(Dropout(0.25))
    model.add(Dense(100))
    model.add(LeakyReLU(0.1))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation = 'sigmoid'))
    model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'], optimizer = 'Adamax')
    return model

# fit model
model = create_model()
history = model.fit(data_input, Y_input, batch_size= 64, epochs = 200, validation_data = 0.2)

# save model
model.save('mnist_model.h5')

#predict test data
y_pred  = model.predict(test_set)
y_output = np.argmax(y_pred, axis = 1)

# save output
df_output = pd.DataFrame()
df_output['prediction'] = y_output
df_output.to_csv('output.csv', index = False,)