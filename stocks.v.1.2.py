import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import linear_model
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot') #specify which style sheet you want

#getting the dataset
df = quandl.get("CHRIS/MGEX_MW1")
#print the head just to get a look at what we are working with
#print(df.head())

#grab certain features in the dataset
#create a list of all the columns to use in the dataset.
df = df[['Open', 'High', 'Low', 'Last', 'Volume']]
#print(df)
#define what features need to be used and get rid of redundant ones.

#firstly doing the high minus the low percent
#define a new column --> 'HL_PCT' = HighLow_Percent
df['HL_PCT'] = (df['High'] - df['Last']) / df['Last'] * 100

#calculate the daily percent change
df['PCT_change'] = (df['Last'] - df['Open']) / df['Open'] * 100

#define a new dataframe with adjusted values
#define only the columns that are needed.
df = df[['Last', 'HL_PCT', 'PCT_change', 'Volume']] #volume refers to the number of trades that were made that day
#print(df.tail)

#features that we have set:'Last', 'HL_PCT', 'PCT_change', 'Volume'
#to define a label, create a new column
forecast_col = 'Last'
df.fillna(-99999, inplace=True)#fill in missing data

#regression alogorithm: 
#define forecast out as being equal to the int() value
#this will predict the number of days out 
forecast_out = int(math.ceil(0.1*len(df))) #ceil() returns the ceiling value of x. math.ceil will round up to the nearest whole
#number of days shifted is 1%
print(forecast_out) #result is 615 days in advanc3

#we will try to predict out 15% of the datafame
#create labels now that we have forecast_out
df ['label'] = df[forecast_col].shift(-forecast_out) #shifting the columns negatively
#each row, which is the label column ie. 'Last' price for each row amount of x days into the future.
#so the features may cause the 'Last' price in x days to change or 1% 
#result of forecast_out is 615 days

#print(df.tail())

#defined features are X, labels will be y
#X is equal to numpy array 
X = np.array(df.drop(['label'], 1)) #drop the label column, therefore the features will be everything but the label column
#dropping 'label' returns a new dataframe
#it can be converted to an array.

X = X[:-forecast_out]
X_lately = X[-forecast_out:] #X_lately is what will actually be used to predict against.
#don't have the y values for X_lately which is why training or testing won't be done on this data.

#scale X before feeding it through the classifier.
#so that it is normalized with the other data points
#to scale it, its included in the training data 
X = preprocessing.scale(X)

#define y
df.dropna(inplace=True)
y = np.array(df['label'])

#print(len(X), len(y))
#pass through the x's, the y's and how big of a test size you want
#this is going to take all the features of the labels
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15)
#We use x_train & y_train to fit our classifier
#use LinearRegression() to define a classifier.
#can now sample a training set while holding out 15% of the data for testing (evaluating) our classifier.

#test using linear regression algorithm
#clf = LinearRegression(n_jobs = -1)
#test using svm algorithm
#clf = svm.SVR()

#clf.fit(X_train, y_train) #fit is synonomous with train
#with open ('linearregression', 'wb') as f:
 #   pickle.dump(clf, f) #saving trained classifier

pickle_in = open('linearregression', 'rb') #to use the classifier
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test) #score is synonomous with test

#print(accuracy)

#Predicting unknown data
#working with the forecast_out shift, which is 615 days, to define our X's
#predict using the x data
forecast_set = clf.predict(X_lately)#can pass an array of values here to make a prediction per value in that array. 
print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan #indicates that thid column contains 'Not A Number' data 

#elements needed to create a graph:
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400 #seconds in a day
next_unix = last_unix + one_day #next_unix is the next day

#iterating through the forecast set, taking each forecast and day and then setting those as the values in the dataframe. 
#Making the future features, nan, okay to use.
#the last line takes all of the first column sets them to numbers and the final column is equal to i, which is the Forecast in this case.
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i] #.loc refrences the index for the dataframe. 'i' is the forecast

print(df.tail())

df['Last'].plot()
df['Forecast'].plot()
plt.legend(loc=4) #bottom right location
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
