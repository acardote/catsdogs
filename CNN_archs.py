
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, AveragePooling2D, MaxPooling2D

a = 12

#Medium post
model_medium = Sequential()

model_medium.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
model_medium.add(MaxPooling2D(pool_size = (2, 2)))
model_medium.add(Flatten())
model_medium.add(Dense(units = 128, activation = 'relu'))
model_medium.add(Dense(units = 1, activation = 'sigmoid'))

#LeNet-5 alg

model_LeNet5 = Sequential()

model_LeNet5.add(Conv2D(filters = 6, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (64,64,3)))
model_LeNet5.add(AveragePooling2D(pool_size=(2,2)))
model_LeNet5.add(Conv2D(filters = 16, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model_LeNet5.add(AveragePooling2D(pool_size=(2,2)))
model_LeNet5.add(Flatten())
model_LeNet5.add(Dense(120, activation = "relu"))
model_LeNet5.add(Dense(84, activation = "relu"))
model_LeNet5.add(Dense(1, activation = "sigmoid"))


#AlexNet
model_AlexNet = Sequential()
model_AlexNet.add(Conv2D(filters = 96, kernel_size = (11,11),strides = 4, 
			  padding = 'Same', activation ='relu', input_shape = (64,64,3)))
model_AlexNet.add(MaxPool2D(pool_size=(3,3), strides = 2))
model_AlexNet.add(Conv2D(filters = 256, kernel_size = (5,5), padding = 'Same', 
                 activation ='relu'))
model_AlexNet.add(MaxPool2D(pool_size=(3,3), strides = 2))
model_AlexNet.add(Conv2D(filters = 384, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model_AlexNet.add(Conv2D(filters = 384, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model_AlexNet.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model_AlexNet.add(MaxPool2D(pool_size=(3,3), strides = 2))
model_AlexNet.add(Flatten())
model_AlexNet.add(Dense(9216, activation = "relu"))
model_AlexNet.add(Dense(4096, activation = "relu"))
model_AlexNet.add(Dense(4096, activation = "relu"))
model_AlexNet.add(Dense(1, activation = "sigmoid"))