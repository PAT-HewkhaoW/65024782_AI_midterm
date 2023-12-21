# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 09:36:00 2023

@author: patch
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier , plot_tree
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

File_Path = 'C:/Users/patch/Downloads/'
File_Name = 'car_data.csv'

df = pd.read_csv(File_Path+File_Name)

df.drop(columns = ['User ID'] , inplace = True)
df.dropna(subset = ['Age','AnnualSalary'] , inplace = True)


encoders = []
enc = LabelEncoder()


for i in range(0 , len(df.columns) - 1 ):
    enc = LabelEncoder()
    df.iloc[: ,i] = enc.fit_transform(df.iloc[: ,i])
    encoders.append(enc)
    
x = df.iloc[:, 0:3]
y = df['Purchased']
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=0)

model = DecisionTreeClassifier(criterion='entropy')
model.fit(x_train , y_train)

x_predict = ['Male' , 50 , 50000]

for i in range(0 , len(df.columns) - 1 ):
    x_predict[i] = encoders[i].transform([x_predict[i]])
    
x_predict_adj = np.array(x_predict).reshape(-1, 3)

y_predict = model.predict(x_predict_adj)
print('Prediction : ' , y_predict[0])
score = model.score(x,y)
print('Accuracy : ' , '{:.2f}'.format(score))

feature = x.columns.tolist()
Data_class = y.tolist()

plt.figure(figsize=(25,20))
_ = plot_tree(model, 
              feature_names = feature, 
              class_names= Data_class,
              label= 'all',
              impurity= True , 
              precision= 3 ,
              filled = True,
              rounded= True,
              fontsize= 18
              )

plt.show()

feature_importances = model.feature_importances_
feature_names = ['Gender' , 'Age' , 'AnnualSalary']
sns.set(rc={'figure.figsize' :(11.7,8)})
sns.barplot(x = feature_importances , y = feature_names)

print(feature_importances)
