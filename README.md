# Project_2_Classification_of_Customers_using_KNN
# Step 1 : Gathering data
import numpy as np # imported numpy
import pandas as pd # imported pandas
data=pd.read_csv("/content/Social_Network_Ads.csv") # uploaded datset
data.head() # displayed first five rows of dataset
df=data.drop(['User ID'],axis=1) # removing unwanted column from dataset
# Step 2 : Data preprocessing
#one hot encoding - will convert the categorical variables into numerical variables
df['Gender']=df['Gender'].apply({'Male':1,'Female':2}.get)
data.head()
# Step 3 : Dividing the data into dependent and independent variables
x=df[['Gender','Age','EstimatedSalary']] #independent variables
y=df['Purchased'] #dependent variables
# Step 4 : Splitting the data into training and testing set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2) # #test_size=0.2 that means 20% of the data will be used for testing the accuracy of the model 
# and 80% of the data will be used for making the model learn
# Step 5 : Creating a ML model using KNN algorithm
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5) # n_neighbors is value of K=5
knn.fit(x_train,y_train) # training - .fit()
y_pred=knn.predict(x_test) # prediction - .predict()
y_pred # for print
# calculating the accuracy of the ML model
from sklearn.metrics import confusion_matrix, accuracy_score
cm=confusion_matrix(y_test,y_pred)
ac=accuracy_score(y_test,y_pred)
print(cm)
print(ac)
