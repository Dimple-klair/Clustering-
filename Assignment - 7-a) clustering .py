#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch


# In[2]:


df=pd.read_csv('C:/Users/RIG1/Desktop/DS ASSIGNMENTS/QUESTIONS -all assignments/ASS 7/crime_data.csv')
df.head()


# # Hierarchical clutering

# In[3]:


# # import inbuilt function for (automatic)standardization
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# scaled_data_df=scaler.fit_transform(df.iloc[:,1:])
# ### so the data has been normlaized 


# In[4]:


# doing standardization using a function

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)


# In[5]:


df_norm=norm_func(df.iloc[:,1:])


# In[6]:


df_norm


# In[7]:


# create dendrogram

den=sch.dendrogram(sch.linkage(df_norm,method='single'))


# In[8]:


# create clusters
# n_cluster = k-value

hc=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='single')


# In[9]:


hc


# In[10]:


# now saving the above cluster for chart purpose

y_hc=hc.fit_predict(df_norm)


# In[11]:


y_hc


# In[12]:


clusters=pd.DataFrame(y_hc,columns=['clusters'])
clusters


# In[13]:


df['h_clusterid']=clusters


# In[14]:


df


# # K-Means

# In[15]:


from sklearn.cluster import KMeans


# In[16]:


df1=pd.read_csv('C:/Users/RIG1/Desktop/DS ASSIGNMENTS/QUESTIONS -all assignments/ASS 7/crime_data.csv')
df1.head()           
                


# In[17]:


# doing standardization using a function

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)


# In[18]:


df_norm1=norm_func(df.iloc[:,1:])
df_norm1


# # now dendrogram
# ### we create dendrogram to create cluster
# # but before making cluster ----- our 1st step is to set k-value
# # so,find out best k-value(using a for-loop)

# In[19]:


#wcss= within clusters sum of square

wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,random_state=40)
    kmeans.fit(df_norm1)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow curv')
plt.xlabel('number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[20]:


# taking k=4


# ## now build cluster (cluster algo)

# In[21]:


model=KMeans(4,random_state=40)
model.fit(df_norm1)


# In[22]:


model.labels_


# In[23]:


# 1= 2nd cluster
#3= 4th cluster
#0th = 1st cluster


# ### ASSIGN CLUSTERS TO DATASET

# In[24]:


df['cluster_new']=model.labels_
#df['cluster_new']


# In[25]:


df


# In[26]:


# as we have only 2 clusters -------------- in hieracrchical 
# here also there r 2 clusters based on -------(h_clusterid)



df.groupby('h_clusterid').agg(['mean']).reset_index()


# ### 0th cluster ==== 1st cluster
# ### 1st ========== 2nd cluster show the places with high murder rates,assault,rape also less population as compared to 1st cluster

# In[27]:


from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# In[29]:


df2=pd.read_csv('C:/Users/RIG1/Desktop/DS ASSIGNMENTS/QUESTIONS -all assignments/ASS 7/crime_data.csv')
df2.head()


# In[30]:


df2.info()


# In[31]:


# as first column is string so we cant apply normalization on it so do this

df3=df2.iloc[:,1:]
df3.head()


# In[38]:


array=df3.values
array


# In[39]:


# now getting out dataframe in ----- array form
## bec, standardized scaler need data in ----- array form


# In[40]:


scale=StandardScaler()
x=scale.fit(array)
x1=x.transform(array)
x1


# ### Now apply DBSCAN algo on above stand. array

# In[46]:


dbscan=DBSCAN(eps=1,min_samples=6)
dbscan.fit(x1)


# ### get the NOISY POINTS
# 
# #### values with -1 is the noisy data/noisy labels/outliers
# 
# 
# 
# ##### inc and dec the eps-val affects the output we get
# ##### if eps-val increases--------outliers decreases
# ##### if eps-val increases--------outliers increases

# In[47]:


dbscan.labels_


# In[48]:


# get it in the dataframe

cl=pd.DataFrame(dbscan.labels_,columns=['clusters'])


# In[49]:


cl


# ### above we have only 1 cluster-----0th
# ### rest is outlier ------------     -1

# In[51]:


pd.concat([df3,cl],axis=1)


# In[ ]:




