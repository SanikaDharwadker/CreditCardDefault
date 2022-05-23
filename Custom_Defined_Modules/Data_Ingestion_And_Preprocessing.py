#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN 
import random
from sklearn.preprocessing import MinMaxScaler
import pickle
import logging


# In[6]:

logging.basicConfig(filename='Data_Ingestion_Preprocessing_Log', level=logging.INFO,
                    format='%(levelname)s:%(asctime)s:%(message)s')


# For Capping Outliers

def cap_upper(df,feature,limit):
    
    percentile_limit=df[feature].quantile(limit)
    df.loc[df[feature]>=percentile_limit,feature]=percentile_limit
    return df[feature]


# In[7]:


# For Handelling Data Imbalance

def desired_class_balance(X_train,y_train,per_majority,per_minority):
    
    
    df_train=pd.concat([X_train,y_train],axis=1)
    df_train.reset_index(drop=True,inplace=True)
    
    ad = ADASYN()
    X_train_a, y_train_a = ad.fit_resample(X_train, y_train)
    df_train_a=pd.concat([X_train_a,y_train_a],axis=1)
    trans_1_a=df_train_a[df_train_a['default.payment.next.month']==1]
    trans_1_a.reset_index(drop=True,inplace=True)
    
    
    num_trans_majority=len(df_train[df_train['default.payment.next.month']==0])
    num_trans_minority=len(df_train[df_train['default.payment.next.month']==1])
    
    desired_ratio=per_majority/per_minority
    
    num_trans_reqd_minor=(num_trans_majority-(num_trans_minority*desired_ratio))/(desired_ratio)
    num_trans_reqd_minor=int(num_trans_reqd_minor)
    
    index=[]
    trans=1

    while trans<=num_trans_reqd_minor:
        ind=random.randint(0, len(trans_1_a)-1)

        if ind not in index:
            index.append(ind)
            trans+=1
            
    for ind in index:
        df_train=df_train.append(trans_1_a.iloc[ind],ignore_index=True)
    
    X_train=df_train.drop(['default.payment.next.month'],axis=1)
    y_train=df_train['default.payment.next.month']
    
    df_train=pd.concat([X_train,y_train],axis=1)
    
    return df_train


# In[ ]:


# To split data into train and test set

def train_test_splitter(df):
    
    X=df.drop(['default.payment.next.month'],axis=1)
    y=df['default.payment.next.month']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y, 
                                                    test_size=0.2)
    return X_train, X_test, y_train, y_test


# In[2]:


def load_data_and_preprocess(path):
    
    df=pd.read_csv(path)
    logging.info("Dropping Id column")
    df.drop('ID',axis=1,inplace=True)
    
    logging.info("Data Processing Unknown Values")
    # Replace 0 by unknown
    df['MARRIAGE']=df['MARRIAGE'].replace({0:3})
    df['EDUCATION']=df['EDUCATION'].replace({0:4,5:4,6:4})
    
    
    df.loc[df['LIMIT_BAL']>=df['LIMIT_BAL'].quantile(0.95),'LIMIT_BAL']=df['LIMIT_BAL'].quantile(0.95)
    

    logging.info("Creating a new feature to capture positivity of Bill_Amt Features")
    for i in range(1,7):
        feature='BILL_AMT'+str(i)+'_POS'
        df[feature]=df['BILL_AMT'+str(i)].apply(lambda x:1 if x>=0 else 0)
    logging.info("New Features Formed")
    
    logging.info("Taking The absolute Of bill amounts")
    for i in range(1,6):
        feature='BILL_AMT'+str(i)
        df[feature]=df[feature].apply(lambda x:abs(x))
       
    logging.info("Outlier Treatment in process")
    for i in range(1,6):
        feature='BILL_AMT'+str(i)
        df[feature]=cap_upper(df,feature,limit=0.95)
        
    # Outlier Treatment
    for i in range(1,7):
        feature='PAY_AMT'+str(i)
        df[feature]=cap_upper(df,feature,limit=0.90)
    logging.info("Outlier Treatment Complete")
    
    logging.info("Splitting Data into train and test sets")
    X_train, X_test, y_train, y_test=train_test_splitter(df)
    
    logging.info("Balancing the data")
    
    df_train_d=desired_class_balance(X_train=X_train,y_train=y_train,per_majority=0.65,per_minority=.35)
    df_test_d=pd.concat([X_test,y_test],axis=1)
    
    return df_train_d,df_test_d


# In[ ]:




