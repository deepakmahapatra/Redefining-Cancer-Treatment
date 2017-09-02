
# In[60]:

from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix


# In[ ]:

variance_list=[]
for i in range(191):
    svd = TruncatedSVD(n_components=i, n_iter=7, random_state=42)
    svd.fit(tfidf_matrix) 
    #print(svd.explained_variance_ratio_)
    #print(svd.explained_variance_ratio_.sum())
    variance_list.append(svd.explained_variance_ratio_.sum()*100)


# In[210]:

import matplotlib.pyplot as plt
plt.plot(variance_list)
plt.ylabel('% Variance Explained')
plt.xlabel('Number of Componenets Selected')
plt.show()


# In[211]:

with open('variance_list.pickle','wb') as f:
		pickle.dump(variance_list,f)


# In[61]:

selected_svd=TruncatedSVD(n_components=500, n_iter=7, random_state=42)
#


# In[62]:

SVD_data=selected_svd.fit_transform(train_tfidf_matrix)
SVD_data=pd.DataFrame(SVD_data)
print SVD_data.shape
SVD_data_test=selected_svd.fit_transform(test_tfidf_matrix)
SVD_data_test=pd.DataFrame(SVD_data_test)
print SVD_data_test.shape


# In[ ]:




# In[63]:

#train_dummies= pd.get_dummies(final, prefix='Category_', columns=['Gene','Variation'])
#test_dummies=pd.get_dummies(final_test, prefix='Category_', columns=['Gene','Variation'])
train_dummies=final
test_dummies=final_test
test_dummies.shape


# In[ ]:




# In[64]:

train_dummies.drop("Text",axis=1,inplace=True)
test_dummies.drop("Text",axis=1,inplace=True)


# In[65]:

print train_dummies.head(2)
print test_dummies.head(2)
train_dummies.drop(["Variation","Gene"],axis=1,inplace=True)
test_dummies.drop(["Variation","Gene"],axis=1,inplace=True)


# In[66]:

train_dummies.head(2)


# In[67]:

print SVD_data.shape
print SVD_data_test.shape
print train_dummies.shape
print test_dummies.shape


# In[68]:

train_SVD=SVD_data.merge(train_dummies,left_index=True,right_index=True)
test_SVD=SVD_data_test.merge(test_dummies,left_index=True,right_index=True)


# In[69]:

print train_SVD.shape
print test_SVD.shape


# In[70]:





# In[73]:

train_SVD.drop(["ID"],axis=1,inplace=True)

test_SVD.drop(["ID"],axis=1,inplace=True)


# In[74]:

train_SVD.Class=train_SVD.Class-1


# In[ ]:




# In[75]:

train_SVD.to_csv("Train500.csv")
test_SVD.to_csv("Test500.csv")


# In[76]:

label_train_SVD=(train_SVD.pop("Class"))
#label_test_SVD=test_SVD.pop("Class")
print label_train_SVD.shape
