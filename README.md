# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.


## Program:

```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: KISHORE V
RegisterNumber:  212224240077

  import pandas as pd
  data=pd.read_csv("Employee.csv")
  print("data.head():")
  data.head()

  
  print("data.info():")
  data.info()

  print("isnull() and sum():")
  data.isnull().sum()

  print("data value counts():")
  data["left"].value_counts()

  from sklearn.preprocessing import LabelEncoder
  le=LabelEncoder()

  print("data.head() for Salary:")
  data["salary"]=le.fit_transform(data["salary"])
  data.head()

  print("x.head():")
                    x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
  x.head()

  y=data["left"]
  from sklearn.model_selection import train_test_split
  x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
  from sklearn.tree import DecisionTreeClassifier
  dt=DecisionTreeClassifier(criterion="entropy")
  dt.fit(x_train,y_train)
  y_pred=dt.predict(x_test)

  print("Accuracy value:")
  from sklearn import metrics
  accuracy=metrics.accuracy_score(y_test,y_pred)
  accuracy

  print("Data Prediction:")
  dt.predict([[0.5,0.8,9,260,6,0,1,2]])

  from sklearn.tree import plot_tree
  import matplotlib.pyplot as plt
  
  plt.figure(figsize=(8,6))
  plot_tree(dt, feature_names=x.columns, class_names=['salary', 'left'], filled=True)
  plt.show()
```

## Output:
<img width="1301" height="282" alt="image" src="https://github.com/user-attachments/assets/509dc1a3-f115-4797-a4ac-95929cb8261a" />

<img width="838" height="462" alt="image" src="https://github.com/user-attachments/assets/8c9e7185-c4dd-4997-8a45-d02536fbd2be" />

<img width="330" height="288" alt="image" src="https://github.com/user-attachments/assets/781875ca-ec22-4940-bd39-c5566369184b" />

<img width="364" height="142" alt="image" src="https://github.com/user-attachments/assets/ecec5ff1-3e9d-45d9-bd48-d5192ab576db" />

<img width="1286" height="230" alt="image" src="https://github.com/user-attachments/assets/a968ad96-b413-4951-8d79-f37823f308bc" />

<img width="1328" height="269" alt="image" src="https://github.com/user-attachments/assets/d335d083-6315-46ec-b979-b268a18af368" />

<img width="1291" height="274" alt="image" src="https://github.com/user-attachments/assets/81d581e0-b1c5-4509-b91d-eda9e4f5cfa9" />

<img width="238" height="67" alt="image" src="https://github.com/user-attachments/assets/92e7eef8-7ae1-43d1-aa1a-b7083b5c2705" />

<img width="211" height="54" alt="image" src="https://github.com/user-attachments/assets/d0177597-7746-4d49-ac1f-41a0250a5944" />




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
