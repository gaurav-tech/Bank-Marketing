# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 19:28:57 2020

@author: GAURAV SHARMA
"""


# import h2ogpu as sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import model_selection
from sklearn import neighbors
from sklearn import naive_bayes
from sklearn import ensemble
from sklearn import tree
from sklearn import feature_selection
from sklearn import metrics
from sklearn import preprocessing
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import re

def analysefeatures(pdf):
    inttypes=['int16','int32','int64']
    floattypes=['float32','float64']
    intcols=pdf.select_dtypes(include=inttypes).columns.values
    floatcols=pdf.select_dtypes(include=floattypes).columns.values
    catcols=pdf.select_dtypes(include=['object']).columns.values
    return intcols,floatcols,catcols

def modelwisetest(modellist,Xtrain,Xtest,Ytrain,Ytest):
    serieslist=[]
    for model in modellist:
        scores=np.zeros(Xtrain.shape[1])
        for i,col in enumerate(Xtrain.columns.values):
             #model=ensemble.RandomForestClassifier(100)
             
             model.fit(Xtrain[col].values.reshape(-1,1),Ytrain)
             predtest=model.predict(Xtest[col].values.reshape(-1,1))
             auc=metrics.roc_auc_score(Ytest,predtest)
             scores[i]=auc
        ser=pd.Series(scores)
        ser.index=Xtrain.columns
        serieslist.append(ser.sort_values(ascending=False))
    return serieslist

def reports(ytrue,predicted):
    print("Accuracy : {}".format(metrics.accuracy_score(ytrue,predicted)))
    print("Precision : {}".format(metrics.precision_score(ytrue,predicted)))
    print("Recall : {}".format(metrics.recall_score(ytrue,predicted)))
    print("F1_score : {}".format(metrics.f1_score(ytrue,predicted)))
    ##print("Logloss : {}".format(metrics.log_loss(ytrue,predicted)))
    print("AUC : {}".format(metrics.roc_auc_score(ytrue,predicted)))

def build(Xtrain,Xtest,ytrain,ytest):
    model=linear_model.LogisticRegression()
    model.fit(Xtrain,ytrain)
    predtest=model.predict(Xtest)
    predtrain=model.predict(Xtrain)
    print("########TRAIN REPORT#########")
    reports(ytrain,predtrain)
    print("########TEST REPORT#########")
    reports(ytest,predtest)
    
def build1(Xtrain,Xtest,ytrain,ytest):
    model=ensemble.RandomForestClassifier(n_estimators=500,max_depth=10,min_samples_split=2,min_samples_leaf=1,random_state=5)
    model.fit(Xtrain,ytrain)
    predtest=model.predict(Xtest)
    predtrain=model.predict(Xtrain)
    print("########TRAIN REPORT#########")
    reports(ytrain,predtrain)
    print("########TEST REPORT#########")
    reports(ytest,predtest)
    
    
def build2(Xtrain,Xtest,ytrain,ytest):
    model=tree.DecisionTreeClassifier(max_depth=10,min_samples_split=2,min_samples_leaf=1,random_state=42)
    model.fit(Xtrain,ytrain)
    predtest=model.predict(Xtest)
    predtrain=model.predict(Xtrain)
    print("########TRAIN REPORT#########")
    reports(ytrain,predtrain)
    print("########TEST REPORT#########")
    reports(ytest,predtest)    
    
#build1(Xtrain_all,Xtest_all,Ytrain_all,Ytest_all)    
def modelstats1(Xtrain,Xtest,ytrain,ytest):
    stats=[]
    modelnames=["LR","DecisionTree","KNN","NB"]
    models=list()
    models.append(linear_model.LogisticRegression())
    models.append(tree.DecisionTreeClassifier())
    models.append(neighbors.KNeighborsClassifier())
    models.append(naive_bayes.GaussianNB())
    for name,model in zip(modelnames,models):
        if name=="KNN":
            k=[l for l in range(5,17,2)]
            grid={"n_neighbors":k}
            grid_obj = model_selection.GridSearchCV(estimator=model,param_grid=grid,scoring="f1")
            grid_fit =grid_obj.fit(Xtrain,ytrain)
            model = grid_fit.best_estimator_
            model.fit(Xtrain,ytrain)
            name=name+"("+str(grid_fit.best_params_["n_neighbors"])+")"
            print(grid_fit.best_params_)
        else:
            model.fit(Xtrain,ytrain)
        trainprediction=model.predict(Xtrain)
        testprediction=model.predict(Xtest)
        scores=list()
        scores.append(name+"-train")
        scores.append(metrics.accuracy_score(ytrain,trainprediction))
        scores.append(metrics.precision_score(ytrain,trainprediction))
        scores.append(metrics.recall_score(ytrain,trainprediction))
        scores.append(metrics.f1_score(ytrain,trainprediction))
        scores.append(metrics.roc_auc_score(ytrain,trainprediction))
        stats.append(scores)
        scores=list()
        scores.append(name+"-test")
        scores.append(metrics.accuracy_score(ytest,testprediction))
        scores.append(metrics.precision_score(ytest,testprediction))
        scores.append(metrics.recall_score(ytest,testprediction))
        scores.append(metrics.f1_score(ytest,testprediction))
        scores.append(metrics.roc_auc_score(ytest,testprediction))
        stats.append(scores)
    
    colnames=["MODELNAME","ACCURACY","PRECISION","RECALL","F1","AUC"]
    return pd.DataFrame(stats,columns=colnames)

df=pd.read_csv('D:\\ML_DATA\\projectdata_machinelearning\\bank_marketing\\bank.csv')
df.head(10)
df=df.replace('\;', ' ',regex=True)
df=df.replace('\"', '',regex=True)
df.head()



tri="""age";"job";"marital";"education";"default";"balance";"housing";"loan";"contact";"day";"month";"duration";"campaign";"pdays";"previous";"poutcome";"y"""
newstr=re.sub('[^a-zA-Z0-9 \n\.]',' ',tri)
newstr
newstr=re.sub(' +',' ',newstr)
newstr=re.sub('"','',newstr)
newstr

df=df.rename(columns={'"age";"job";"marital";"education";"default";"balance";"housing";"loan";"contact";"day";"month";"duration";"campaign";"pdays";"previous";"poutcome";"y"' : 'age job marital education default balance housing loan contact day month duration campaign pdays previous poutcome y' })



n_df=pd.DataFrame(df['age job marital education default balance housing loan contact day month duration campaign pdays previous poutcome y'].str.split(' ',16).tolist(),columns=['age' ,'job', 'marital' ,'education', 'default', 'balance' ,'housing' ,'loan', 'contact' ,'day', 'month' ,'duration', 'campaign' ,'pdays', 'previous', 'poutcome' ,'y'])

n_df["y"]=n_df.y.map(dict(yes=1, no=0))
n_df[["age","balance","day","duration","campaign", "pdays", "previous","y"]] = n_df[["age","balance","day","duration","campaign", "pdays", "previous","y"]].apply(pd.to_numeric)

intcols,floatcols,catcols=analysefeatures(n_df)
intcols

print(n_df.dtypes)

######### cat_df #############
catcols
cat_df=pd.DataFrame(n_df[catcols])



y_df=pd.DataFrame(n_df["y"])
# Categorical boolean mask
categorical_feature_mask = cat_df.dtypes==object
# filter categorical columns using mask and turn it into a list
categorical_cols = cat_df.columns[categorical_feature_mask].tolist()
categorical_cols

cat_df=cat_df.apply(LabelEncoder().fit_transform)

obj=feature_selection.SelectKBest(score_func=feature_selection.f_classif,k=5) 
obj.fit(cat_df,y_df)      
cat_df.columns.values[obj.get_support()]

lst=['education', 'housing', 'loan', 'contact', 'poutcome']     
arr=np.array(lst)

fcat_df=cat_df[arr]

Xtrain_c,Xtest_c,Ytrain_c,Ytest_c=model_selection.train_test_split(fcat_df,y_df,test_size=.2,random_state=0)
build1(Xtrain_c,Xtest_c,Ytrain_c,Ytest_c)
build(Xtrain_c,Xtest_c,Ytrain_c,Ytest_c)
modelstats1(Xtrain_c,Xtest_c,Ytrain_c,Ytest_c)




########## int_df ###############
intcols
int_df=pd.DataFrame(n_df[intcols])
int_df=int_df.drop("y",axis=1)
obj=feature_selection.SelectKBest(score_func=feature_selection.f_classif,k=5) 
obj.fit(int_df,y_df)      
int_df.columns.values[obj.get_support()]

lst=['age', 'duration', 'campaign', 'pdays', 'previous']     
arr=np.array(lst)

fint_df=int_df[arr]


y_df=pd.DataFrame(fint_df["y"])
int_df=int_df.drop(int_df["y"])
Xtrain_i,Xtest_i,Ytrain_i,Ytest_i=model_selection.train_test_split(fint_df,y_df,test_size=.2,random_state=0)
build1(Xtrain_i,Xtest_i,Ytrain_i,Ytest_i)
build(Xtrain_i,Xtest_i,Ytrain_i,Ytest_i)
modelstats1(Xtrain_i,Xtest_i,Ytrain_i,Ytest_i)


############concatting int and categorical cols ###################

f_df=pd.concat([fint_df,fcat_df],axis=1)
#f_df=f_df.dropna(axis=0)
Xtrain_f,Xtest_f,Ytrain_f,Ytest_f=model_selection.train_test_split(f_df,y_df,test_size=.2,random_state=0)
build1(Xtrain_f,Xtest_f,Ytrain_f,Ytest_f)
build(Xtrain_f,Xtest_f,Ytrain_f,Ytest_f)
modelstats1(Xtrain_f,Xtest_f,Ytrain_f,Ytest_f)





obj=feature_selection.SelectKBest(score_func=feature_selection.f_classif,k=7) 
obj.fit(f_df,y_df)
f_df.columns.values[obj.get_support()]


fvalue,probability=feature_selection.f_classif(f_df,y_df)
ser=pd.Series(probability)
ser.index=f_df.columns
ser[:7].sort_values(ascending=False).plot.bar()
ser.sort_values(ascending=False,inplace=True)
ser.plot.bar(rot=0)
ser1=pd.Series(fvalue)
ser1.index=f_df.columns
ser1[:7].sort_values(ascending=False).plot.bar()
ser1.sort_values(ascending=False,inplace=True)
obj=feature_selection.SelectKBest(score_func=feature_selection.f_classif,k=7) 
obj.fit(f_df,y_df)      
f_df.columns.values[obj.get_support()]



lst=['duration', 'pdays', 'previous', 'housing', 'loan', 'contact','poutcome']     
       
arr=np.array(lst)

X_final=f_df[arr]

Xtrain_fin,Xtest_fin,Ytrain_fin,Ytest_fin=model_selection.train_test_split(X_final,y_df,test_size=.22,random_state=0)
build1(Xtrain_fin,Xtest_fin,Ytrain_fin,Ytest_fin)
########TRAIN REPORT#########
Accuracy : 0.9741917186613727
Precision : 0.9904153354632588
Recall : 0.7788944723618091
F1_score : 0.8720112517580871
AUC : 0.8889676965389608
########TEST REPORT#########
Accuracy : 0.8954773869346734
Precision : 0.6021505376344086
Recall : 0.45528455284552843
F1_score : 0.5185185185185186
AUC : 0.706426680092489


build(Xtrain_fin,Xtest_fin,Ytrain_fin,Ytest_fin)
modelstats1(Xtrain_fin,Xtest_fin,Ytrain_fin,Ytest_fin)

build2(Xtrain_fin,Xtest_fin,Ytrain_fin,Ytest_fin)
