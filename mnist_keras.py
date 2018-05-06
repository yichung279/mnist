import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation,Convolution2D,Flatten
from keras.optimizers import  Adam
from tensorflow.examples.tutorials.mnist import input_data

def build_model():
    #建立模型
    model = Sequential()
    #將模型疊起
    model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='same', input_shape=(28, 28, 1), activation='relu') 
    model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='same',  activation='relu') 
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=128, kernel_size=(5, 5), padding='same',  activation='relu') 
    model.add(Conv2D(filters=128, kernel_size=(5, 5), padding='same',  activation='relu') 
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=256, kernel_size=(5, 5), padding='same',  activation='relu') 
    model.add(Conv2D(filters=256, kernel_size=(5, 5), padding='same',  activation='relu') 
    model.add(Conv2D(filters=256, kernel_size=(5, 5), padding='same',  activation='relu') 
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=512, kernel_size=(5, 5), padding='same',  activation='relu') 
    model.add(Conv2D(filters=512, kernel_size=(5, 5), padding='same',  activation='relu') 
    model.add(Conv2D(filters=512, kernel_size=(5, 5), padding='same',  activation='relu') 
    model.add(Conv2D(filters=512, kernel_size=(5, 5), padding='same',  activation='relu') 
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=512, kernel_size=(5, 5), padding='same',  activation='relu') 
    model.add(Conv2D(filters=512, kernel_size=(5, 5), padding='same',  activation='relu') 
    model.add(Conv2D(filters=512, kernel_size=(5, 5), padding='same',  activation='relu') 
    model.add(Conv2D(filters=512, kernel_size=(5, 5), padding='same',  activation='relu') 
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    mode.add(Flatten())
    model.add(Dense(units=4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=10,activation='softmax'))
    model.summary()
    return model
if __name__ == '__main__':
    mnist = input_data.read_data__sets("MNIST_data/", onr_hot=true)
    x_tyrain, y_train= mnist.train.images, mnist.train.labels  
    x_test, y_test = mnist.test.images, mnist.test.labels  
    model = build_model()
    #開始訓練模型
    model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
    model.fit(x_train,y_train,batch_size=100,epochs=20)
    #顯示訓練結果
    score = model.evaluate(x_train,y_train)
    print ('\nTrain Acc:', score[1])
    score = model.evaluate(x_test,y_test)
    print ('\nTest Acc:', score[1])
