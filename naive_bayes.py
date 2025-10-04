#-------------------------------------------------------------------------
# AUTHOR: Aaron Hamm
# FILENAME: naive_bayes
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd

dbTraining = []
dbTest = []

#Reading the training data using Pandas
df = pd.read_csv('weather_training.csv')
for _, row in df.iterrows():
    dbTraining.append(row.tolist())

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
outlook_map = {'Sunny': 1, 'Overcast': 2, 'Rain': 3}
temp_map    = {'Hot': 1, 'Mild': 2, 'Cool': 3}
hum_map     = {'High': 1, 'Normal': 2}
wind_map    = {'Weak': 1, 'Strong': 2}

X = []
for r in dbTraining:
    X.append([outlook_map[r[1]],temp_map[r[2]],hum_map[r[3]],wind_map[r[4]]])
#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
Y = [1 if r[5] == 'Yes' else 2 for r in dbTraining]

#Fitting the naive bayes to the data using smoothing
#--> add your Python code here
clf = GaussianNB()
clf.fit(X,Y)

#Reading the test data using Pandas
df = pd.read_csv('weather_test.csv')
for _, row in df.iterrows():
    dbTest.append(row.tolist())

#Printing the header of the solution
#--> add your Python code here
print(f"{'Day':<6} {'Outlook':<10} {'Temperature':<10} {'Humidity':<10} {'Wind':<10} {'PlayTennis':<12} {'Confidence':<10}")

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
for r in dbTest:
    x_test = [[
        outlook_map[r[1]],
        temp_map[r[2]],
        hum_map[r[3]],
        wind_map[r[4]]
    ]]
    proba = clf.predict_proba(x_test)[0]
    best_idx = proba.argmax()
    confidence = proba[best_idx]
    prediction_class_num = clf.classes_[best_idx]
    if prediction_class_num == 1:
        prediction_label = 'Yes'
    else:
        prediction_label = 'No'

    if confidence >= 0.75:
        print(f"{r[0]:<6} {r[1]:<10} {r[2]:<10} {r[3]:<10} {r[4]:<10} {prediction_label:<12} {confidence:<.2f}")