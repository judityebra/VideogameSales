#!/usr/bin/env python
# coding: utf-8

# Judit Yebra Valencia (1603614)

# # VIDEO GAMES SALES WITH RATINGS

# In[1]:


import math
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import svm, datasets
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict,GridSearchCV,StratifiedKFold,LeaveOneOut,cross_val_score,train_test_split
from sklearn.metrics import mean_absolute_error,roc_curve,auc,roc_auc_score,f1_score,precision_recall_curve,classification_report,confusion_matrix
import time

# import some data to play with
dataset = pd.read_csv("C:/Users/Judit/Desktop/uab/tercer/APC/VIDEOGAMES_ind/Video_Games_Sales_as_at_22_Dec_2016.csv", header=0, delimiter=',')


# In[2]:


dataset.head() #es veu de quin tipus són les columnes


# In[26]:


dataset.info() #es veuen les columnes que té el dataset inicialment


# In[3]:


print("Dimensionalitat de la BBDD:", dataset.shape)


# Un cop es té la dimensionalitat de la base de dades se li fa una còpia per poder netejar-la bé amb més comoditat.

# In[4]:


dataset_nou = dataset.copy()


# In[5]:


#S'eliminen les següents columnes ja que no són necessaries: 
dataset_nou = dataset_nou.drop(['Name', 'Year_of_Release', 'Critic_Count', 'User_Count'], axis=1)


# In[6]:


dataset.Platform.unique()


# Es fa un OneHotEncoder a la columna categòrica que proporciona la informació de les plataformes i es transforma en diverses columnes binàries.

# In[7]:


platforms = OneHotEncoder(handle_unknown='ignore')
platforms_df = pd.DataFrame(platforms.fit_transform(dataset_nou[['Platform']]).toarray())
dataset_nou = dataset_nou.join(platforms_df)


# In[8]:


dataset_nou.head()


# In[9]:


dataset_nou.info()


# In[10]:


dataset_nou.loc[dataset_nou[30] == 1][['Platform',30]] #es local·litza quin número és i se li fa la seva pròpia columna


# In[11]:


dataset_nou.head() #es pot observar com hi ha unes columnes 0...30 on cadascuna es refereix a una plataforma


# In[12]:


dataset_nou = dataset_nou.rename({0: '2600', 1: '3DO', 2: '3DS', 3: 'DC', 4: 'DS', 5: 'GB', 6: 'GBA', 7: 'GC', 8: 'GEN', 9: 'GG', 10: 'N64', 11: 'NES', 12: 'NG', 13: 'PC', 14: 'PCFX', 15: 'PS', 16: 'PS2', 17: 'PS3', 18: 'PS4', 19: 'PSP', 20: 'PSV', 21: 'SAT', 22: 'SCD', 23: 'SNES', 24: 'TG16', 25: 'WS', 26: 'Wii', 27:'WiiU', 28: 'X360', 29: 'XB', 30: 'XOne'}, axis='columns')


# S'han cambiat el nom de les columnes pel nom de cada plataforma per transformar un valor categòric que no ens era de cap utlitat en un binari.

# In[13]:


print("Per visualitzar les primeres 5 mostres de la base de dades i veure com han cambiat els números per les plataformes:")
dataset_nou.head()


# In[14]:


#ja es pot eliminar la columna de les plataformes
dataset_nou.drop('Platform', axis ='columns', inplace=True)


# In[15]:


dataset_nou.head() #ja no està la columna categòrica platforms


# La següent columna categòrica a la qual haurem de fer un OneHotEncoder és la dels gèneres del joc.

# In[16]:


dataset.Genre.unique()


# In[17]:


genre = OneHotEncoder(handle_unknown='ignore')
genre_df = pd.DataFrame(genre.fit_transform(dataset_nou[['Genre']]).toarray())
dataset_nou = dataset_nou.join(genre_df)


# In[18]:


dataset_nou.head() #tenim noves columnes amb nombres generades pel OneHotEncoder


# In[19]:


dataset_nou.info()
#es pot observar com de la columna 46 a la 58 tenim els nombres als quals els hi hem de donar el nom d'un gènere


# In[20]:


dataset_nou = dataset_nou.rename({0: 'Sports', 1: 'Platform', 2: 'Racing', 3: 'Role-playing', 4: 'Puzzle', 5: 'Misc', 6: 'Shooter', 7: 'Simulation', 8: 'Action', 9: 'Fighting', 10: 'Adventure', 11: 'Strategy', 12: 'No Genre'}, axis='columns')


# In[21]:


print("Per visualitzar les primeres 5 mostres de la base de dades i veure com han cambiat els números pels gèneres:")
dataset_nou.head()


# In[22]:


#ja es pot eliminar la columna del gènere
dataset_nou.drop('Genre', axis ='columns', inplace=True)


# In[23]:


#comprovem que s'ha esborrat
dataset_nou.head()


# In[40]:


dataset.Rating.unique() #es miren quants tipus hi ha i es fa un OneHotEncoder


# In[41]:


rating = OneHotEncoder(handle_unknown='ignore')
rating_df = pd.DataFrame(rating.fit_transform(dataset_nou[['Rating']]).toarray())
dataset_nou = dataset_nou.join(rating_df)


# In[42]:


dataset_nou.info()


# In[43]:


dataset_nou = dataset_nou.rename({0: 'E', 1: 'NC', 2: 'M', 3: 'T', 4: 'E10+', 5: 'K-A', 6: 'AO', 7: 'EC', 8: 'RP'}, axis='columns')


# In[44]:


#ja es pot eliminar la columna 
dataset_nou.drop('Rating', axis ='columns', inplace=True)


# In[45]:


#comprovem que s'ha esborrat
dataset_nou.head()


# In[24]:


dataset.Publisher.unique() #no podem fer OneHotEncoder massa columnes


# In[27]:


dataset.Developer.unique()


# Com es pot observar les columnes 'Publisher' i 'Developer' estan conformades per moltíssims valors unique(), per tant no es pot fer un OneHotEncoder perquè quedarien moltíssimes columnes. Es farà per tant un MeanEncoding on s'agafaran les ventes promig de cadascuna de les categories de publisher i developer i es substituiran en la columa de publisher i developer.

# In[33]:


publisher = dataset_nou.groupby(['Publisher'])['NA_Sales'].mean().to_dict()
dataset_nou['Publisher'] = dataset_nou['Publisher'].map(publisher)


# In[36]:


publisher = dataset_nou.groupby(['Developer'])['NA_Sales'].mean().to_dict()
dataset_nou['Developer'] = dataset_nou['Developer'].map(publisher)


# In[47]:


dataset_nou.head() #ja tenim números a la columna de Publisher i de Developer


# Un cop s'han eliminat les columnes categòriques, es fa l'eliminació de 
# NaNs, es poden observar a continuació on hi ha i quants:

# In[46]:


dataset_nou.info()


# In[ ]:





# In[ ]:





# In[ ]:




