# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load Data – Import the employee dataset with relevant features and churn labels.

2. Preprocess Data – Handle missing values, encode categorical features, and split into train/test sets.

3. Initialize Model – Create a DecisionTreeClassifier with desired parameters.

4. Train Model – Fit the model on the training data.

5. Evaluate Model – Predict on test data and check accuracy, precision, recall, etc.

6. Visualize & Interpret – Visualize the tree and identify key features influencing churn. 
 

## Program and Output:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: J.JANANI
RegisterNumber:  212223230085
*/
```

```
import pandas as pd
data = pd.read_csv("C:\\Users\\admin\\OneDrive\\Desktop\\Folders\\ML\\DATASET-20250226\\Employee.csv")
data.head()
```
![436833656-d984b2fb-9820-45a7-b3ef-b550c903a6e6](https://github.com/user-attachments/assets/567758eb-0f85-441b-ade9-678ddc444918)

```
data.info()
data.isnull().sum()
data['left'].value_counts()
```

![436834372-76506a59-7213-43b1-b66e-79eac79e56d5](https://github.com/user-attachments/assets/d5001b6d-4eb6-413f-afa8-f6e9bc630313)

```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['salary'] = le.fit_transform(data['salary'])
data.head()
```

![436834611-f9549631-2268-4346-b72a-16cd323c3b82](https://github.com/user-attachments/assets/9bc66997-20e2-4dc7-b3b5-5e3171c16592)

```
x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()
```

![436834902-1dbe360b-ef38-4ee6-a05d-ea9155785591](https://github.com/user-attachments/assets/eb7b31b2-d2ea-45a4-b3aa-ebfda6acb53a)

```
y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state =100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy
```

![436835250-acf94cbf-afd2-4b7a-aca3-a441556a0cfb](https://github.com/user-attachments/assets/fe3d2d63-08d6-4dd3-9936-e3f2dd580a56)

```
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

![436835575-864f75e5-8279-4e5e-b37e-e646bcaed25a](https://github.com/user-attachments/assets/e0b69b05-af3f-4043-b94f-5fbcb82b7e2b)


```
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree # Import the plot_tree function

plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```
![436835823-2264fa2a-c89c-432f-b772-2a5b2aa8c3a9](https://github.com/user-attachments/assets/f76a887e-0cfa-484b-b7a9-4fe5bc89ec36)










## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
