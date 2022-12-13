#!/usr/bin/env python
# coding: utf-8

# Judit Yebra Valencia (1603614)

# # VIDEO GAMES SALES WITH RATINGS

# In[1]:


import math
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import svm, datasets
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict,GridSearchCV,StratifiedKFold,LeaveOneOut,cross_val_score,train_test_split
from sklearn.metrics import mean_absolute_error,roc_curve,auc,roc_auc_score,f1_score,precision_recall_curve,classification_report,confusion_matrix
from sklearn.linear_model import Lasso
import time

# import some data to play with
dataset = pd.read_csv("C:/Users/Judit/Desktop/uab/tercer/APC/VIDEOGAMES_ind/Video_Games_Sales_as_at_22_Dec_2016.csv", header=0, delimiter=',')


# In[2]:


dataset.head() #es veu de quin tipus són les columnes


# In[3]:


dataset.info() #es veuen les columnes que té el dataset inicialment


# In[4]:


print("Dimensionalitat de la BBDD:", dataset.shape)


# Un cop es té la dimensionalitat de la base de dades se li fa una còpia per poder netejar-la bé amb més comoditat.

# In[5]:


dataset_nou = dataset.copy()


# In[6]:


#S'eliminen les següents columnes ja que no són necessaries: 
dataset_nou = dataset_nou.drop(['Year_of_Release', 'Critic_Count', 'User_Count'], axis=1)


# In[7]:


dataset.Platform.unique()


# Es fa un OneHotEncoder a la columna categòrica que proporciona la informació de les plataformes i es transforma en diverses columnes binàries.

# In[8]:


platforms = OneHotEncoder(handle_unknown='ignore')
platforms_df = pd.DataFrame(platforms.fit_transform(dataset_nou[['Platform']]).toarray())
dataset_nou = dataset_nou.join(platforms_df)


# In[9]:


dataset_nou.head()


# In[10]:


dataset_nou.info()


# In[11]:


dataset_nou.loc[dataset_nou[30] == 1][['Platform',30]] #es local·litza quin número és i se li fa la seva pròpia columna


# In[12]:


dataset_nou.head() #es pot observar com hi ha unes columnes 0...30 on cadascuna es refereix a una plataforma


# In[13]:


dataset_nou = dataset_nou.rename({0: '2600', 1: '3DO', 2: '3DS', 3: 'DC', 4: 'DS', 5: 'GB', 6: 'GBA', 7: 'GC', 8: 'GEN', 9: 'GG', 10: 'N64', 11: 'NES', 12: 'NG', 13: 'PC', 14: 'PCFX', 15: 'PS', 16: 'PS2', 17: 'PS3', 18: 'PS4', 19: 'PSP', 20: 'PSV', 21: 'SAT', 22: 'SCD', 23: 'SNES', 24: 'TG16', 25: 'WS', 26: 'Wii', 27:'WiiU', 28: 'X360', 29: 'XB', 30: 'XOne'}, axis='columns')


# S'han cambiat el nom de les columnes pel nom de cada plataforma per transformar un valor categòric que no ens era de cap utlitat en un binari.

# In[14]:


print("Per visualitzar les primeres 5 mostres de la base de dades i veure com han cambiat els números per les plataformes:")
dataset_nou.head()


# In[15]:


#ja es pot eliminar la columna de les plataformes
dataset_nou.drop('Platform', axis ='columns', inplace=True)


# In[16]:


dataset_nou.head() #ja no està la columna categòrica platforms


# La següent columna categòrica a la qual haurem de fer un OneHotEncoder és la dels gèneres del joc.

# In[17]:


dataset.Genre.unique()


# In[18]:


genre = OneHotEncoder(handle_unknown='ignore')
genre_df = pd.DataFrame(genre.fit_transform(dataset_nou[['Genre']]).toarray())
dataset_nou = dataset_nou.join(genre_df)


# In[19]:


dataset_nou.head() #tenim noves columnes amb nombres generades pel OneHotEncoder


# In[20]:


dataset_nou.info()
#es pot observar com de la columna 46 a la 58 tenim els nombres als quals els hi hem de donar el nom d'un gènere


# In[21]:


dataset_nou = dataset_nou.rename({0: 'Sports', 1: 'Platform', 2: 'Racing', 3: 'Role-playing', 4: 'Puzzle', 5: 'Misc', 6: 'Shooter', 7: 'Simulation', 8: 'Action', 9: 'Fighting', 10: 'Adventure', 11: 'Strategy', 12: 'No Genre'}, axis='columns')


# In[22]:


print("Per visualitzar les primeres 5 mostres de la base de dades i veure com han cambiat els números pels gèneres:")
dataset_nou.head()


# In[23]:


#ja es pot eliminar la columna del gènere
dataset_nou.drop('Genre', axis ='columns', inplace=True)


# In[24]:


#comprovem que s'ha esborrat
dataset_nou.head()


# In[25]:


dataset.Rating.unique() #es miren quants tipus hi ha i es fa un OneHotEncoder


# In[26]:


rating = OneHotEncoder(handle_unknown='ignore')
rating_df = pd.DataFrame(rating.fit_transform(dataset_nou[['Rating']]).toarray())
dataset_nou = dataset_nou.join(rating_df)


# In[27]:


dataset_nou.info()


# In[28]:


dataset_nou = dataset_nou.rename({0: 'E', 1: 'NC', 2: 'M', 3: 'T', 4: 'E10+', 5: 'K-A', 6: 'AO', 7: 'EC', 8: 'RP'}, axis='columns')


# In[29]:


#ja es pot eliminar la columna 
dataset_nou.drop('Rating', axis ='columns', inplace=True)


# In[30]:


#comprovem que s'ha esborrat
dataset_nou.head()


# In[31]:


dataset_nou.isnull().sum()


# In[32]:


dataset.Publisher.unique() #no es pot fer OneHotEncoder massa columnes


# In[33]:


dataset.Developer.unique()


# Com es pot observar les columnes 'Publisher' i 'Developer' estan conformades per moltíssims valors unique(), per tant no es pot fer un OneHotEncoder perquè quedarien moltíssimes columnes. Es farà per tant un MeanEncoding on s'agafaran les ventes promig de cadascuna de les categories de publisher i developer i es substituiran en la columa de publisher i developer.

# In[34]:


publisher = dataset_nou.groupby(['Publisher'])['NA_Sales'].mean().to_dict()
dataset_nou['Publisher'] = dataset_nou['Publisher'].map(publisher)


# In[35]:


publisher = dataset_nou.groupby(['Developer'])['NA_Sales'].mean().to_dict()
dataset_nou['Developer'] = dataset_nou['Developer'].map(publisher)


# In[36]:


dataset_nou.head() #ja tenim números a la columna de Publisher i de Developer


# In[37]:


dataset_nou.loc[dataset_nou['User_Score'] == 'tbd']


# In[38]:


dataset_nou['User_Score'] = dataset_nou['User_Score'].replace(['tbd'], -20)


# In[39]:


#S'eliminen les files que tenen més de 3 nulls en les següents columnes ja que dificulten la regressió
llista = []
for i in range(len(dataset_nou)):
    t = 0
    if(math.isnan(dataset_nou.iloc[i]['Publisher'])):
        t += 1
    if(math.isnan(dataset_nou.iloc[i]['Critic_Score'])):
        t += 1
    try:
        dataset_nou.loc[i,['User_Score']] = float(dataset_nou.iloc[i]['User_Score'])
    except:
        if(math.isnan(dataset_nou.iloc[i]['User_Score'])):
            t += 1
    if(math.isnan(dataset_nou.iloc[i]['Developer'])):
        t += 1
    if(t >= 3):
        llista.append(i)
for i in llista:
    dataset_nou.drop(i, axis = 0, inplace = True)

#Búsqueda manual dels 'Publishers' que falten
for i in range(len(dataset_nou)):
    if(math.isnan(dataset_nou.iloc[i]['Publisher'])): 
        print(dataset_nou.iloc[i]['Name'])
# Els publishers que falten són: 
# 
# Sonic the Hedgehog: SEGA 
# Yu Yu Hakusho: Dark Tournament: Atari 
# Stronghold 3: SouthPeak Games 
# Farming Simulator 2011: Giants Software 
# World of Tanks: Wargaming
# Demolition Company: Gold Edition : Giants Software 
# Dream Dancer : Zoo games
# Homeworld Remastered Collection : Sierra Estudios
# Brothers in Arms: Furious 4 : UBISOFT

# In[40]:


dataset_nou.loc[16718]


# In[ ]:





# In[46]:


#S'afegeixen les dades trobades al dataset
llista = ['SEGA', 'Atari', 'SouthPeak Games', 'Giants Software', 'Wargaming', 'Giants Software', 'Zoo games', 'Sierra Estudios', 'UBISOFT']
llista2 = []
#cont = 0
for j,i in enumerate(dataset_nou['Publisher']):
    if(math.isnan(i)): 
        llista2.append(j)
llista2
        #index = dataset_nou.index[dataset_nou.iloc[j]]
        #dataset_nou.loc[index,['Publisher']] = llista[cont]
        #cont += 1
        #dataset_nou.iloc[j]


# In[49]:


dataset_nou.loc[dataset_nou['Name'] == 'Sonic the Hedgehog']


# In[50]:


dataset_nou.loc[4127,['Publisher']] = 'SEGA'


# In[ ]:


#hacer con los demas


# In[ ]:


dataset_nou.iloc[15402]


# In[51]:


dataset_nou.info()  #veiem 


# Un cop s'han eliminat les columnes categòriques, es fa l'eliminació de 
# NaNs, es poden observar a continuació on hi ha i quants:

# In[ ]:


dataset_nou.info() #amb dataset_nou.isnull().sum() no es veuen totes les columnes 


# In[ ]:





# In[ ]:


dataset_nou.info()
len(dataset_nou)  #es pot observar que s'han eliminat uns 40-50 nulls


# In[ ]:


y = dataset_nou['Publisher']
X = dataset_nou.copy()
X.drop('Publisher', axis ='columns', inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


X_sense_G_train_FS = StandardScaler().fit_transform(X_sense_G_train_FS)


# In[ ]:


cp = dataset_nou.copy()
cp = cp.dropna()
y = cp['Critic_Score']
X = cp.copy()
X.drop(['Critic_Score','User_Score'],  axis ='columns', inplace=True)
#X.drop(['Critic_Score','Publisher','User_Score','Developer'], axis ='columns', inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


X_train = StandardScaler().fit_transform(X_train)


# In[ ]:


Lasso = Lasso()


# In[ ]:


cerca_alpha_lasso = GridSearchCV(Lasso, {'alpha':np.arange(0.1,10,0.1)}, cv = 5, scoring="neg_mean_squared_error")


# In[ ]:


cerca_alpha_lasso.fit(X_train, y_train.values.ravel())


# In[ ]:


print("Resultats Grid Search: \n")
print("Millor estimador dels paràmetres buscats: \n", cerca_alpha_lasso.best_estimator_)
print("Millor score dels paràmetres: \n", cerca_alpha_lasso.best_score_)
print("Millors paràmetres: \n", cerca_alpha_lasso.best_params_)


# In[ ]:


coefs = cerca_alpha_lasso.best_estimator_.coef_
importancia = np.abs(coefs)
importancia


# In[ ]:


X= cp[np.array(X.columns)[importancia > 0].tolist()]


# In[ ]:


X.info()


# In[ ]:


dataset_nou.info()


# In[ ]:


regressor_CS = LinearRegression()
X = StandardScaler().fit_transform(X)
regressor_CS.fit(X,y)


# In[ ]:


Pred_X = dataset_nou.copy()
Pred_X.drop(['Critic_Score', 'User_Score'], axis ='columns', inplace=True)
Pred_X = Pred_X[np.array(Pred_X.columns)[importancia > 0].tolist()]
Pred_X = Pred_X[dataset_nou.isna().any(axis = 1)]
#Pred_X = Pred_X.dropna()


# In[ ]:


Pred_X.info()


# In[ ]:


prediccio_CS = regressor_CS.predict(Pred_X)


# In[ ]:





# In[ ]:





# In[ ]:




