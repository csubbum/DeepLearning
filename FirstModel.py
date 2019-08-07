#!/usr/bin/env python
# coding: utf-8

# In[33]:


from keras.models import Sequential
from keras.layers import Dense
import numpy

seed = 7
numpy.random.seed(seed)


# In[42]:


from pandas import read_csv
filename = "Downloads/KerasDownloads/BBCN.csv"
dataframe = read_csv(filename)
array = dataframe.values
x = array[:,0:11]
y=array[:,11]


# In[43]:


dataframe.head()


# In[44]:


model = Sequential()
model.add(Dense(11, input_dim=11, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))


# In[45]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[46]:


model.fit(x,y,epochs=50,batch_size=10)


# In[47]:


scores=model.evaluate(x,y)
print("%s: %.2f%%" % (model.metrics_names[1],scores[1]*100))


# In[ ]:




