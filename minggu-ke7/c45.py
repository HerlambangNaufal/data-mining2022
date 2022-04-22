#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
from sklearn import tree


# In[7]:


##membaca dataset dari file ke pandas dataFrame
irisDataset =pd.read_csv('Downloads\klasifikasi_dataset_iris.csv',
                         delimiter =',',header=0)


# In[8]:


#mengubah kelas (kolom "species") dari string ke unique-integer
irisDataset["Species"]= pd.factorize(irisDataset.Species[0])
#Menghapus kolom "Id"
irisDataset= irisDataset.drop(labels="Id", axis=1)


# In[9]:


#mengubah dataframe ke array numpy
#irisDataset = irisDataset.as_matrix()
irisDataset = irisDataset.to_numpy()


# In[12]:


#Membagi Dataset, 40 Baris data untuk training
#dan 20 baris data untuk testing
dataTraining = np.concatenate((irisDataset[0:40,:],
                              irisDataset[50:90,:]), axis =0)
dataTesting = np.concatenate((irisDataset[40:50,:],
                              irisDataset[90:100,:]), axis =0)


# In[15]:


#memecah dataset ke input dan label
inputTraining = dataTraining[:,0:4]
inputTesting= dataTesting[:,0:4]
labelTraining = dataTraining[:,4]
labelTraining = dataTesting[:, 4]


# In[16]:


#mendefinisikan decision tree classifier
model = tree.DecisionTreeClassifier()
#mentraining model
model = model.fit(inputTraining, labelTraining)
#memprediksi input data testing
hasilPrediksi = model.predict(inputTesting)
print("label sebenarnya", labelTesting)
print("hasil prediksi: ", hasilPrediksi)
#menghitung akurasi
prediksiBenar =(hasilPrediksi == labelTesting.sum)
prediksiSalah =(hasilPrediksi != labelTesting.sum)
print("prediksi benar: ", prediksiBenar, " data")
print("prediksi salah: ", prediksiSalah, " data")
print("akurasi: ", prediksiBenar/(prediksiBenar+prediksiSalah)
     * 100, "%")


# In[ ]:




