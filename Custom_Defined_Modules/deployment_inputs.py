#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pickle


# In[2]:


def get_inputs_for_model_tme(input_dict):
    
    train_data=pd.read_csv('train.csv')
    train_data.drop(['Unnamed: 0'],axis=1,inplace=True)
    
    target_encoder_dict=pickle.load(open('target_encoder_dict.pickle','rb'))
    
    input_df=pd.DataFrame(input_dict,index=[0])
    
    for i in range(1,7):
        feature='BILL_AMT'+str(i)+'_POS'
        input_df[feature]=input_df['BILL_AMT'+str(i)].apply(lambda x:1 if x>=0 else 0)
        
    for i in range(1,6):
        feature='BILL_AMT'+str(i)
        input_df[feature]=input_df[feature].apply(lambda x:abs(x))
        
    
    
    categoric_columns=['SEX','EDUCATION','MARRIAGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']

    for col in target_encoder_dict.keys():
        replace_dict=target_encoder_dict[col]
        input_df[col]=input_df[col].replace(replace_dict)
        
    train_data.drop(['default.payment.next.month'],axis=1,inplace=True)
    input_df=input_df[train_data.columns]
            
    

    
    inputs=input_df.iloc[0].to_list()
    
    return inputs


def get_inputs_for_model_ohe(input_dict):
    
    train_data=pd.read_csv('train.csv')
    train_data.drop(['default.payment.next.month','Unnamed: 0'],axis=1,inplace=True)
    
    input_df=pd.DataFrame(columns=train_data.columns)
    
    train_cols=list(train_data.columns)
    pred_x={}
    
    
    numeric_cols=['AGE',
       'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5',
       'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5',
       'PAY_AMT6',]

    for col in train_cols:
        if col not in pred_x.keys():
            pred_x[col]=0
            
    for col in numeric_cols:
        pred_x[col]=input_dict[col]
        
    
        
    binary_vars=['SEX','MARRIAGE','EDUCATION','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']


    for bv in binary_vars:
        search_feature=bv+'_'+input_dict[bv]
    
        if search_feature in pred_x:
            pred_x[search_feature]=1
        else:
            continue
            
    input_df=input_df.append(pred_x,ignore_index=True)
    
    for i in range(1,7):
        feature='BILL_AMT'+str(i)+'_POS'
        input_df[feature]=input_df['BILL_AMT'+str(i)].apply(lambda x:1 if x>=0 else 0)
        
    for i in range(1,6):
        feature='BILL_AMT'+str(i)
        input_df[feature]=input_df[feature].apply(lambda x:abs(x))
        
     
    inputs=input_df.iloc[0].to_list()
    
    return inputs
    
    

