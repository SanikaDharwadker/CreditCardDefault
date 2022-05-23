#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from imblearn.ensemble import BalancedRandomForestClassifier


# In[6]:


def get_xy_traintest(scale,scaler):
    
    
    if (scale==True) & (scaler=='MinMax') :
        
        df_train=pd.read_csv('train_scaled.csv')
        df_test=pd.read_csv('test_scaled.csv')
        
        df_test.drop('Unnamed: 0',axis=1,inplace=True)
        df_train.drop('Unnamed: 0',axis=1,inplace=True)
        
        
        
        X_train=df_train.drop('default.payment.next.month',axis=1)
        y_train=df_train['default.payment.next.month']

        X_test=df_test.drop('default.payment.next.month',axis=1)
        y_test=df_test['default.payment.next.month']
    
    
    elif (scale==True) & (scaler=='StandardScaler'):
        
        df_train=pd.read_csv('train.csv')
        df_test=pd.read_csv('test.csv')
        
        df_test.drop('Unnamed: 0',axis=1,inplace=True)
        df_train.drop('Unnamed: 0',axis=1,inplace=True)

        X_train=df_train.drop('default.payment.next.month',axis=1)
        y_train=df_train['default.payment.next.month']

        X_test=df_test.drop('default.payment.next.month',axis=1)
        y_test=df_test['default.payment.next.month']
        
        
        scaler=StandardScaler()
        X_train=pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns,index=X_train.index)
        X_test=pd.DataFrame(scaler.transform(X_test),columns=X_test.columns,index=X_test.index)
        
    else:
        
        df_train=pd.read_csv('train.csv')
        df_test=pd.read_csv('test.csv')
        
        df_test.drop('Unnamed: 0',axis=1,inplace=True)
        df_train.drop('Unnamed: 0',axis=1,inplace=True)
        
        X_train=df_train.drop('default.payment.next.month',axis=1)
        y_train=df_train['default.payment.next.month']

        X_test=df_test.drop('default.payment.next.month',axis=1)
        y_test=df_test['default.payment.next.month']
        
        
    return X_train,y_train,X_test,y_test


        
    


# In[8]:


def build_basic_model(X_train,y_train,X_test,y_test,classifier):   
    
    classifiers={'Logistic Regression':LogisticRegression(max_iter=500,random_state=42),'Random Forest':RandomForestClassifier(random_state=42),'Ada boost':AdaBoostClassifier(random_state=42),'GBOOST':GradientBoostingClassifier(),'XGBOOST':XGBClassifier()}
    
    
    classifier_basic=classifiers[classifier]
    
    classifier_basic=classifier_basic.fit(X_train,y_train)
    
    y_test_prob=classifier_basic.predict_proba(X_test)[:,1]
    y_train_prob=classifier_basic.predict_proba(X_train)[:,1]
    
    
    roc_auc_score_test=roc_auc_score(np.array(y_test),y_test_prob)
    roc_auc_score_train=roc_auc_score(np.array(y_train),y_train_prob)
    
    scores=pd.DataFrame(columns=['MODEL','PARAMS','y_train_prob','y_test_prob','TRAIN SCORE','TEST SCORE','DIFFERENCE'])
    scores=scores.append({'MODEL':"Basic",'PARAMS':classifier_basic,'y_train_prob':y_train_prob,'y_test_prob':y_test_prob,'TRAIN SCORE':roc_auc_score_train,'TEST SCORE':roc_auc_score_test,'DIFFERENCE':roc_auc_score_train-roc_auc_score_test},ignore_index=True)
    
    return scores
    
    


# In[10]:


def store_model_predictions(model_name,y_test_prob):
    
    
    store_model_pred=pd.DataFrame()
    store_model_pred[model_name]=y_test_prob 
    
    store_model_pred.to_csv('{0}_Predictions'.format(model_name))
    
# In[11]:
    
def build_model(X_train,y_train,X_test,y_test,classifier,score_df,classifier_name):   
    
    classifier=classifier.fit(X_train,y_train)
    
    y_test_prob=classifier.predict_proba(X_test)[:,1]
    y_train_prob=classifier.predict_proba(X_train)[:,1]
    
    
    roc_auc_score_test=roc_auc_score(np.array(y_test),y_test_prob)
    roc_auc_score_train=roc_auc_score(np.array(y_train),y_train_prob)
    
    score_df=score_df.append({'MODEL':classifier_name,'TRAIN SCORE':roc_auc_score_train,'TEST SCORE':roc_auc_score_test,'DIFFERENCE':roc_auc_score_train-roc_auc_score_test,'PARAMS':classifier,'y_train_prob':y_train_prob,'y_test_prob':y_test_prob},ignore_index=True)
    
    return score_df


def build_ensemble(X_train,y_train,X_test,y_test,score_df,classifier_name):
    
    
    sum_y_train_prob=0
    sum_y_test_prob=0
    for i in range(0,len(score_df)):    

        sum_y_train_prob+=score_df.iloc[i]['y_train_prob']
        sum_y_test_prob+=score_df.iloc[i]['y_test_prob']

    y_train_prob=sum_y_train_prob/len(score_df)
    y_test_prob=sum_y_test_prob/len(score_df)

    train_score=roc_auc_score(y_train,y_train_prob)
    test_score=roc_auc_score(y_test,y_test_prob)

    score_df=score_df.append({'MODEL':classifier_name+'_ensemble','y_train_prob':y_train_prob,'y_test_prob':y_test_prob,'TRAIN SCORE':train_score,'TEST SCORE':test_score,'DIFFERENCE':train_score-test_score},ignore_index=True)
    
    return score_df









