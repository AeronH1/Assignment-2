#-------------------------------------------------------------------------
# AUTHOR: Aaron Hamm
# FILENAME: decision_tree_2.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn import tree
import pandas as pd

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

#Reading the test data in a csv file using pandas
dbTest = []
df_test = pd.read_csv('contact_lens_test.csv')
for _, row in df_test.iterrows():
    dbTest.append(row.tolist())

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #Reading the training data in a csv file using pandas
    # --> add your Python code here
    df_train = pd.read_csv(ds)
    for _, row in df_train.iterrows():
        dbTraining.append(row.tolist())

    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
    age_map = {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3}
    spectacle_map = {'Myope': 1, 'Hypermetrope': 2}
    astig_map = {'No': 1, 'Yes': 2}
    tear_map = {'Reduced': 1, 'Normal': 2}
    class_map = {'Yes': 1, 'No': 2}
    
    for r in dbTraining:
        X.append([age_map[r[0]], 
                  spectacle_map[r[1]], 
                  astig_map[r[2]], 
                  tear_map[r[3]]])
    #Transform the original categorical training classes to numbers and add to the vector Y.
    #For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    for r in dbTraining:
        Y.append(class_map[r[4]])
    #Loop your training and test tasks 10 times here
    accuracies = []
    for i in range (10):
       # fitting the decision tree to the data using entropy as your impurity measure and maximum depth = 5
       # --> addd your Python code here
       # clf =
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
        clf.fit(X, Y)
       #Read the test data and add this data to dbTest
       #--> add your Python code here
        #already read at beginning
        correct = 0
        for data in dbTest:
           #Transform the features of the test instances to numbers following the same strategy done during training,
           #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
           #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           #--> add your Python code here
            feature = [age_map[data[0]], spectacle_map[data[1]], astig_map[data[2]], tear_map[data[3]]]
            class_predicted = clf.predict([feature])[0]
           #Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
           #--> add your Python code here
            true_label = class_map[data[4]]
            if class_predicted == true_label:
                correct += 1
                
        accuracies.append(correct/len(dbTest))
        
    #Find the average of this model during the 10 runs (training and test set)
    #--> add your Python code here
    avg_acc = sum(accuracies)/len(accuracies)

    #Print the average accuracy of this model during the 10 runs (training and test set).
    #Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    print(f'final accuracy when training on {ds}: {avg_acc}')