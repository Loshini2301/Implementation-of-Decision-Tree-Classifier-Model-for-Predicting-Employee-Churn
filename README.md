# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn
### NAME : LOSHINI.G
### DEPARTMENT :IT
### REFERENCE NUMBER: 212223220051

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas.

2.Calculate the null values from dataframe and apply label encoder.

3.Apply decision tree classifier on the dataframe.

4.obtain the value of accuracy and data prediction.

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: LOSHINI.G
RegisterNumber:  212223220051

import pandas as pd
data=pd.read_csv("/content/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
### Initial dataset:
![Screenshot 2024-04-02 083927](https://github.com/Loshini2301/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/150007305/bb028d69-33f3-4be9-ac05-463f5176fec6)


### Data info:

![Screenshot 2024-04-02 084007](https://github.com/Loshini2301/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/150007305/76296688-ba33-4037-8ec2-6560ee848857)

### Null values:
![Screenshot 2024-04-02 084039](https://github.com/Loshini2301/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/150007305/b79322b2-3326-4c47-a5d2-bbc73ef1d2ab)


### Assignment of x and y values:
![Screenshot 2024-04-02 084114](https://github.com/Loshini2301/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/150007305/9f2f76d9-5fc2-4878-a54f-6db481dc0e40)
![Screenshot 2024-04-02 084145](https://github.com/Loshini2301/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/150007305/129c8836-178a-48c0-b740-522dc59d2691)


### Converting string literals to numerical values using label encode
![Screenshot 2024-04-02 084209](https://github.com/Loshini2301/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/150007305/46163e16-378f-415a-9761-f4952cd92abe)

### Accuracy:
![Screenshot 2024-04-02 084313](https://github.com/Loshini2301/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/150007305/c2d74a36-c8a4-443a-b4e8-743eeb5ac9b0)


### Prediction:
![Screenshot 2024-04-02 084421](https://github.com/Loshini2301/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/150007305/2f49409d-396e-4cd5-b783-634a4287061a)


## Result:
Thus the program to implement the Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
