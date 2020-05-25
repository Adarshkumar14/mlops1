#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.datasets import mnist


# In[ ]:


dataset=mnist.load_data('mymnist.db')


# In[ ]:


len(dataset)


# In[ ]:


import keras


# In[ ]:


train,test=dataset


# In[ ]:


len(train)


# In[ ]:


X_train,y_train=train
X_test,y_test=test


# In[ ]:


X_train.shape


# In[ ]:


y_test.shape[0]


# In[ ]:


X_train_1d=X_train.reshape(-1,28,28,1)
X_test_1d=X_test.reshape(-1,28,28,1)


# In[ ]:


X_train_1d.shape


# In[ ]:


X_train = X_train_1d.astype('float32')
X_test = X_test_1d.astype('float32')


# In[ ]:


y_train.shape


# In[ ]:


from keras.utils.np_utils import to_categorical


# In[ ]:


X_train /= 255
X_test /= 255


# In[ ]:


y_train=to_categorical(y_train)
y_test=to_categorical(y_test)


# In[ ]:


y_train


# In[ ]:


y_test.shape[1]


# In[ ]:


from keras.models import Sequential


# In[ ]:


from keras.layers import Dense


# In[ ]:


from keras.layers import Convolution2D


# In[ ]:


from keras.layers import MaxPooling2D


# In[ ]:


from keras.layers import Flatten


# In[ ]:


model=Sequential()


# In[ ]:


model.add(Convolution2D(filters=20,
                       kernel_size=(3,3),
                       activation='relu',
                        input_shape=(28,28,1)
                       ))


# In[ ]:


model.add(MaxPooling2D(pool_size=(2, 2)))


# In[ ]:


model.summary()


# In[ ]:


model.add(Flatten())


# In[ ]:


model.summary()


# In[ ]:



model.add(Dense(units=512,activation='relu'))


# In[ ]:


model.add(Dense(units=128,activation='relu'))


# In[ ]:


model.add(Dense(units=32,activation='relu'))


# In[ ]:


model.add(Dense(units=10,activation='softmax'))


# In[ ]:


x=model.layers[1].output.shape[1:3]


# In[ ]:


import numpy as np
print(x)
y=(29,29)
y=np.array(y)
if (x < y).any():
    print('true')


# In[ ]:


#from keras.optimizers import adam


# In[ ]:


model.compile(optimizer=keras.optimizers.Adadelta(), loss='categorical_crossentropy', 
             metrics=['accuracy']
             )


# In[ ]:


history = model.fit(X_train, y_train,
        
          epochs=2,
          validation_data=(X_test, y_test),
          shuffle=True)


# In[ ]:


y_pred=model.predict(X_test)


# In[ ]:


scores = model.evaluate(X_test,y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


# In[ ]:


model.save("/root/mnist_model.h5")


# In[ ]:


file1 = open("/root/result.txt","w")


# In[ ]:


file1.write(str(scores[1]*100))
file1.close()
