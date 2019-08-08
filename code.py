import pandas as pd
import numpy as np
import random as rnd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def printGeneralStatistics( data ):
 print("\n\n-----------------DATA ANALYSIS-----------------\n")
 print("-------Description of training dataset--------\n")
 print("--------------Statistical Data-------------\n")
 print( data.describe(),"\n" )
 print("------------Distribution Data--------------\n")
 print( data.describe(include=['O']) )
 print("\n")

def printGeneralInformation( data ):
 print("---------Feature//Variable Names-----------\n\n", data.columns.values ,"\n")
 print( data.info,"\n" )

def pivotingData ( data, entry1, entry2, groupBy, sortBy ):
 return data[[ entry1 , entry2 ]].groupby([groupBy], as_index=False).mean().sort_values(by=sortBy, ascending=False)

def printPivotedData( data ):
 print  ("------------------Sex--------------------")
 print (pivotingData( train_df, 'Sex', 'Survived', 'Sex', 'Survived' ),"\n")
 print  ("------------------Pclass--------------------")
 print (pivotingData( train_df, 'Pclass', 'Survived', 'Pclass', 'Survived' ),'\n')
 print  ("------------------Age--------------------")
 print (pivotingData( train_df, 'Age', 'Survived', 'Age', 'Survived' ),'\n')
 print  ("------------------Fare--------------------")
 print (pivotingData( train_df, 'Fare', 'Survived', 'Fare', 'Survived' ),'\n')

def visualizeNumericalCorrelation(): # data, feature1, feature2 ):
 #fig=plt.figure()
 grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
 grid.map(plt.hist, 'Age', alpha=.5, bins=20)
 grid.add_legend();
 grid.savefig("output.png")
 plt.show()
def visualizeScatter(): # data, feature1, feature2 ):
 #sns.set(style="darkgrid")
 grid = sns.FacetGrid(train_df, row='Embarked', size=3, aspect=2)
 grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette="Set2", order=None, hue_order=None )
 grid.add_legend()
 grid.savefig("output2.png")
 plt.show()
def visualizeSurvivedCorrelation(): #  feature1, feature2 ):
 survived = 'survived'
 not_survived = 'not survived'
 fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
 women = train_df[train_df['Sex']=='female']
 men = train_df[train_df['Sex']=='male']
 ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=40, label = survived, ax = axes[0], kde =False)
 ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
 ax.legend()
 ax.set_title('Female')
 ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=40, label = survived, ax = axes[1], kde = False)
 ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)
 ax.legend()
 ax.set_title('Male')
 plt.savefig("output3.png")
 plt.show()



def normalizeSex ( ):  
 for dataset in combine:
   dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

def normalizeAges ( ):
 for dataset in combine:
    mean = train_df["Age"].mean()
    std = test_df["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    # compute random numbers between the mean, std 
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    # fill random values generated in the NaN values in Age column
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = dataset["Age"].astype(int)

def setAgeBoundaries (  ):
 for dataset in combine:
     dataset.loc[ dataset['Age'] <= 5, 'Age'] = 0
     dataset.loc[(dataset['Age'] > 5 ) & (dataset['Age'] <= 16), 'Age'] = 1
     dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 2
     dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 3
     dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 4
     dataset.loc[ dataset['Age'] > 64, 'Age'] = 5

def normalizeFamily( ):
 for dataset in combine:
   dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1


def normalizeEmbarked( ):
 freq_port = train_df.Embarked.dropna().mode()[0]

 for dataset in combine:
   dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

 for dataset in combine:
   dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

def normalizeFare():

 for dataset in combine:
   dataset.loc[(dataset['Fare'] < 9), 'Fare'] = 0
   dataset.loc[(dataset['Fare'] >= 9) & (dataset ['Fare'] < 12), 'Fare'] = 1
   dataset.loc[(dataset['Fare'] >= 12) & (dataset ['Fare'] < 15), 'Fare'] = 2
   dataset.loc[(dataset['Fare'] >= 15) & (dataset ['Fare'] < 20), 'Fare'] = 3
   dataset.loc[(dataset['Fare'] >= 20) & (dataset ['Fare'] < 30), 'Fare'] = 4
   dataset.loc[(dataset['Fare'] >= 30) & (dataset ['Fare'] < 55), 'Fare'] = 5
   dataset.loc[(dataset['Fare'] >= 55) & (dataset ['Fare'] < 95), 'Fare'] = 6
   dataset.loc[(dataset['Fare'] >= 95),'Fare'] = 7
   dataset['Fare'] = dataset['Fare'].astype(int)



def normalizeAgeClass( ):
 for dataset in combine:
   dataset['Age*Class*Fare'] = dataset.Age * dataset.Pclass * dataset.Fare
   dataset['Age*Class'] = dataset.Age * dataset.Pclass
   dataset['Age*Fare'] = dataset.Age * dataset.Fare


def normalizeData( ):
 normalizeSex ( )
 normalizeAges( )
 setAgeBoundaries( )
 normalizeFamily( )
 normalizeEmbarked( )
 normalizeFare( )
 normalizeAgeClass( )

def classifyWithLogisticRegression ( trainingData, results, testData ):
 logreg = LogisticRegression()
 logreg.fit(trainingData, results)
 acc_log = round(logreg.score(trainingData, results) * 100, 2)
 return logreg.predict(testData),acc_log


def classifyWithSVM ( trainingData, results, testData ):
 svm = SVC()
 svm.fit(trainingData,results)
 acc_svm=round(svm.score(trainingData, results) * 100, 2)
 return svm.predict(testData),acc_svm


def classifyWithStochasticGradientDescent ( trainingData, results, testData ):
 sgd = SGDClassifier(max_iter=40, tol=None)
 sgd.fit(trainingData, results)
 acc_sgd = round(sgd.score(trainingData, results) * 100, 2)
 return sgd.predict(testData),acc_sgd


def classifyWithRandomForest ( trainingData, results, testData ):
 random_forest = RandomForestClassifier(n_estimators=100)
 random_forest.fit(trainingData, results)
 acc_random_forest = round(random_forest.score(trainingData, results) * 100, 2)
 return random_forest.predict(testData) , acc_random_forest



def main ( ):
 global train_df
 global test_df
 global combine

 # Reading from CSV
 train_df = pd.read_csv('train.csv')
 test_df = pd.read_csv('test.csv')

 visualizeNumericalCorrelation() 
 visualizeScatter()
 visualizeSurvivedCorrelation()
 

 printGeneralStatistics(train_df) 
  

# Drop Useless Features
 train_df = train_df.drop(['Ticket', 'Cabin', 'Name'], axis=1)
 test_df = test_df.drop(['Ticket', 'Cabin', 'Name'], axis=1)


 test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

 # Normalize both data sets
 combine = [train_df, test_df]
 normalizeData( )
 train_df = train_df.drop(['Parch', 'SibSp'], axis=1)
 test_df = test_df.drop(['Parch', 'SibSp'], axis=1)

 printPivotedData(train_df)
 

 combine = [train_df, test_df]
 printGeneralInformation(train_df)

 # Setting up data
 X_train = train_df.drop(["Survived","PassengerId","Fare","Age","Pclass"], axis=1)
 Y_train = train_df["Survived"]
 X_test  = test_df.drop(["PassengerId","Fare","Age","Pclass"], axis=1).copy()
 X_train.shape, Y_train.shape, X_test.shape

 print ("-----X Training dataset values-----\n\n",X_train,"\n")

 # calling learning algorithms
 [prediction1,acc_log] = classifyWithLogisticRegression(X_train, Y_train, X_test)
 [prediction2,acc_svm] = classifyWithSVM(X_train, Y_train, X_test)
 [prediction3,acc_sgd] = classifyWithStochasticGradientDescent(X_train, Y_train, X_test)
 [prediction4,acc_random_forest] = classifyWithRandomForest(X_train, Y_train, X_test) 
 
 #printing scores of learning algorithms
 results = pd.DataFrame({
    'Model': ['Logistic Regression','Support Vector Machine','Stochastic Gradient Decent', 'Random Forest'],
    'Score': [acc_log,acc_svm,acc_sgd,acc_random_forest]})
 result_df = results.sort_values(by='Score', ascending=False)
 result_df = result_df.set_index('Score')
 print(result_df.head())

 #Build the answer
 submission = pd.DataFrame({
   "PassengerId": test_df["PassengerId"],
   "Survived": prediction4
   })

 # Put it in csv file
 submission.to_csv('submission.csv', index=False)

 print("\n \n Since Random forest has the highest score of all the learning algorithms ,its output will be written to the output CSV file\n\n")

main( )
