#!/usr/bin/env python
# coding: utf-8

# In[16]:



from keras.datasets import mnist
dataset=mnist.load_data('mymnist.db')
len(dataset)
import keras
train,test=dataset
len(train)

X_train,y_train=train
X_test,y_test=test

X_train.shape

y_test.shape[0]

X_train_1d=X_train.reshape(-1,28,28,1)
X_test_1d=X_test.reshape(-1,28,28,1)

X_train_1d.shape

X_train = X_train_1d.astype('float32')
X_test = X_test_1d.astype('float32')

y_train.shape

from keras.utils.np_utils import to_categorical

X_train /= 255
X_test /= 255

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)


from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Convolution2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten
temp=0
neuron=500
out_l=y_test.shape[1]




import numpy as np
min=(1,1)
min=np.array(min)
count=0


# In[17]:



def add_l(top_model):

    top_model=(Convolution2D(filters=20,
                       kernel_size=(3,3),
                       activation='relu',
                       
                            ))(top_model)
    top_model=(MaxPooling2D(pool_size=(2, 2)))(top_model)
    return top_model


# In[18]:


def l_model(n,out,count):
    
    model=Sequential()
    
    model.add(Convolution2D(filters=20,
                       kernel_size=(3,3),
                       activation='relu',
                        input_shape=(28,28,1)
                       ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    print(count)
    temp=count
    i=1
    from  keras.models import Model
    top_model=model.output
    model.summary()
    if count > 0 :
            
        while temp != 0 :
            top_model=(add_l(top_model))
            model=Model(inputs=model.input,outputs=top_model)
            top_model=model.output
            
            if(model.output.shape[1:3] == min ):
                break
            temp=temp-1
            
    nmodel=model.output
    flat = Flatten()(nmodel)

    nmodel=(Dense(units=n,activation='relu'))(flat)
    n=int(n/2)
    nmodel=(Dense(units=n,activation='relu'))(nmodel)
    n=int(n/2)
    nmodel=(Dense(units=n,activation='relu'))(nmodel)

    nmodel=(Dense(units=out,activation='softmax'))(nmodel)
    
    model=Model(inputs=model.input,outputs=nmodel)
    #from keras.optimizers import adam
    model.summary()
    model.compile(optimizer=keras.optimizers.Adadelta(), loss='categorical_crossentropy', 
             metrics=['accuracy']
             )
    return model


# In[19]:


def resetWeights():
    print("Reseting weights")
    w = model.get_weights()
    w = [[j*0 for j in i] for i in w]
    
    model.set_weights(w)


# In[21]:


def t_model():
    model.fit(X_train, y_train,epochs=1,
          validation_data=(X_test, y_test),
          shuffle=True)
    scores = model.evaluate(X_test,y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    accuracy =scores[1]*100
    return accuracy


# In[22]:


accuracy=0.0

best_accuracy=accuracy
best_neuron=0
best_count=0
while accuracy < 99 and count <5:
    
    neuron=neuron+count*2
    model=l_model(neuron,out_l,count)
    count=count+1
    accuracy=t_model()
    if best_accuracy < accuracy:
        best_accuracy=accuracy
        best_neuron=neuron
    resetWeights()   
model=l_model(best_neuron,out_l,best_count)
best_accuracy=t_model()
model.save('/root/mnist_update.h5')
print("model saved")
file1=open("/root/result.txt","w")
file1.write(str(best_accuracy))
file1.close()


# In[ ]:





# In[ ]:




