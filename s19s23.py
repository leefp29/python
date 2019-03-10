# https://www.datacamp.com/community/tutorials/convolutional-neural-networks-python
#from keras.datasets import fashion_mnist
#(train_X,train_Y), (test_X,test_Y) = fashion_mnist.load_data()
import numpy as np
import pandas as pd
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import scipy.io as spio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # 标准化
from sklearn.preprocessing import MinMaxScaler
import pandas, sys
import pandas as pd
import numpy as np
from keras.models import load_model

#matplotlib inline

mnist_loc = 'F:/lfp1224/03092'


data_digits1 =   np.loadtxt(mnist_loc + "/S23FP4-784-6000-train.csv",delimiter = ",")
data_digits2 =   np.loadtxt(mnist_loc + "/S19FP4-784-6000-train.csv",delimiter = ",")
data_digits3 =   np.loadtxt(mnist_loc + "/S23FP4-784-2000-test.csv",delimiter = ",")
data_digits4 =   np.loadtxt(mnist_loc + "/S19FP4-784-2000-test.csv",delimiter = ",")

#data_digits5 =   np.loadtxt(mnist_loc + "/S21FP4-784-197000-test.csv",delimiter = ",")
#data_digits6 =   np.loadtxt(mnist_loc + "/S23FP4-784-200008-test.csv",delimiter = ",")
#data_digits5 =   np.loadtxt(mnist_loc + "/S23free-784-200002-test.csv",delimiter = ",")
#data_digits6 =   np.loadtxt(mnist_loc + "/S21free-784-200001-test.csv",delimiter = ",")

#data_digits5 =   np.loadtxt(mnist_loc + "/S21FP2-784-100001-test.csv",delimiter = ",")
#data_digits6 =   np.loadtxt(mnist_loc + "/S23FP2-784-100001-test.csv",delimiter = ",")

#data_digits2 =   np.loadtxt(mnist_loc + "/S23free-784-600001-all50.csv",delimiter = ",")
#data_digits1 =   np.loadtxt(mnist_loc + "/S21free-784-600001-all50.csv",delimiter = ",")
#data_digits3 =   np.loadtxt(mnist_loc + "/S21FP4-784-600000-all50.csv",delimiter = ",")
#data_digits4 =   np.loadtxt(mnist_loc + "/S23FP4-784-600001-all50.csv",delimiter = ",")


#data_digits5 =   np.loadtxt(mnist_loc + "/S21FP2-784-200001-all50-parttest.csv",delimiter = ",")
#data_digits6 =   np.loadtxt(mnist_loc + "/S23FP2-784-200001-all50-parttest.csv",delimiter = ",")


train_X1 =  data_digits1[:,0:-1]
train_X2 =  data_digits2[:,0:-1]
train_X3 =  data_digits3[:,0:-1]
train_X4 =  data_digits4[:,0:-1]


#train_X5 =  data_digits5[:,0:-1]
#train_X6 =  data_digits6[:,0:-1]
#train_Y5 =  data_digits5[:,-1].astype(int)
#train_Y6 =  data_digits6[:,-1].astype(int)



train_Y1 =  data_digits1[:,-1].astype(int)
train_Y2 =  data_digits2[:,-1].astype(int)
train_Y3 =  data_digits3[:,-1].astype(int)
train_Y4 =  data_digits4[:,-1].astype(int)


print(train_X1.shape,train_X2.shape,train_Y1.shape,train_Y2.shape)
'''
train_X1 = train_X1.reshape(-1,28,28, 1)
train_X2 = train_X2.reshape(-1,28,28, 1)
train_X3 = train_X3.reshape(-1,28,28, 1)
train_X4 = train_X4.reshape(-1,28,28, 1)
'''


train_X=np.concatenate((train_X1,train_X2))

test_X=np.concatenate((train_X3,train_X4))




train_X = train_X.astype('float32')
train_X  = train_X /train_X.sum(axis=1, keepdims=True)
train_X = train_X.reshape(-1,28,28, 1)

test_X = test_X.astype('float32')
test_X  = test_X /test_X.sum(axis=1, keepdims=True)
test_X = test_X.reshape(-1,28,28, 1)


train_Y=np.concatenate((train_Y1,train_Y2))

test_Y=np.concatenate((train_Y3,train_Y4))

#print(train_X)
#print(train_Y)

print('Training data shape : ', train_X.shape, train_Y.shape)

classes = np.unique(train_Y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

print(train_X.shape,train_Y.shape)



# Display the change for category label using one-hot encoding
print('Original label:', train_Y[0])
print('After conversion to one-hot:', train_Y_one_hot[0])
from sklearn.model_selection import train_test_split
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.1, random_state=193)
print(train_X.shape,valid_X.shape,train_label.shape,valid_label.shape)
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.advanced_activations import ReLU
from keras.callbacks import ModelCheckpoint
from keras import regularizers

batch_size = 128
epochs = 2000
num_classes = 4

L2=0.000001
fashion_model = Sequential()

fashion_model.add(Conv2D(16, kernel_size=(4, 4),kernel_initializer='he_normal',padding='same', kernel_regularizer=regularizers.l1(L2),input_shape=(28,28,1)))

fashion_model.add(Conv2D(32, kernel_size=(4, 4),padding='same',input_shape=(28,28,1)))
fashion_model.add(BatchNormalization(axis=2))
fashion_model.add(LeakyReLU())
fashion_model.add(Dropout(0.00001))
fashion_model.add(AveragePooling2D(pool_size=(2, 2),padding='same'))








#fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
#fashion_model.add(LeakyPReLU(alpha=0.1))                  
#fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
#fashion_model.add(Dropout(0.4))
fashion_model.add(Flatten())
fashion_model.add(Dense(128))
#fashion_model.add(BatchNormalization(axis=1))
fashion_model.add(LeakyReLU())           
fashion_model.add(Dropout(0.00001))
fashion_model.add(Dense(num_classes, activation='softplus'))


fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=1e-4),metrics=['accuracy'])

fashion_model.summary()

#fashion_train_dropout = fashion_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=2,validation_data=(valid_X, valid_label))

checkpointer = keras.callbacks.ModelCheckpoint('test-1115-cnnr567-e1500.h5', monitor='val_acc', verbose=2, save_best_only=True, save_weights_only=False, mode='max', period=1)

class TestCallback(keras.callbacks.Callback):
   def __init__(self, test_data):
       self.test_data = test_data

   def on_epoch_end(self, epoch, logs={}):
       x, y = self.test_data
       loss, acc = self.model.evaluate(x, y, verbose=0)
       print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

#history = model.fit(train_x, y_train, validation_data=(x_valid,y_valid), epochs=epoch, batch_size=batch_size, callbacks=[TestCallback((x_test, y_test)),checkpointer])#,callbacks = [checkpointer])

#fashion_train_dropout = fashion_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=2,validation_data=(valid_X, valid_label),callbacks=[TestCallback((test_X,test_Y_one_hot)),checkpointer])


fashion_train_dropout = fashion_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=2,validation_data=(test_X,test_Y_one_hot),callbacks=[TestCallback((valid_X, valid_label)),checkpointer])


fashion_model.save("fashion-kr567-e1500-1115.h5py")

accuracy = fashion_train_dropout.history['acc']
val_accuracy = fashion_train_dropout.history['val_acc']
loss = fashion_train_dropout.history['loss']
val_loss = fashion_train_dropout.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'b', label='Training accuracy')#o 代表散点，b蓝色，r红色，不带o线型图
plt.title('Training accuracy')
plt.legend()
plt.savefig(fname="accuracy-cnnr567-e1500.png",dpi=100)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.title('Training loss')
plt.legend()
plt.savefig(fname="loss-cnnr567-e1500.png",dpi=100)

plt.show()
'''
hist = load_model('test-0905-k444-all82.h5')               
test_loss, test_accuracy = hist.evaluate(test_X,test_Y_one_hot)
print('\n test loss: ', test_loss)
print('test accuracy: ', test_accuracy)
'''
