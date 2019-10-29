# Regression
# A simple Machine Learning example using linear regression.
Regression using python's quandl function

following the tutorial of sentdex https://www.youtube.com/watch?v=JcI5Vnw0b2c&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v&index=2

continued...

Now we will be predicting unknown data
X's defined by working with the forecast_out shift, which is 615 days.

The missing data is dropped so that when creating the labels we have both the X's and X_lately values.
X_lately is what will actually be used to predict against.
X_lately is the 615 days which was calculated.
Don't have the y values for X_lately which is why training or testing won't be done on this data.

Equation of a straight line: y = mx + b
getting the answer for y means that we have done linear regression.

print(forecast_set, accuracy, forecast_out) - result is the unknown values in the dataset for the next 615 day. ie. The stock prices.
Graph these results using matplotlib.

To find out what the last date was, use last_date = df.iloc[-1].name
When doing a prediction, have to specify the date you are doing the prediction for.

So when doing Machine Learning, x and y does not necessarily correspond to the axes on a graph.
In this case, x (the features) and y (the labels), y is the price but x is not correct as the date is not a feature.

Having the dates means we can now populate the dataframe with the new dates and the forecast values.

The red line plotted on the graph represents the known data, whereas the blue indicates the unknown or forecasted data. 

The for loop:
[np.nan for _ in range(len(df.columns) - 1)], that is a list of values that are np.nan
'Last', 'HL_PCT', 'PCT_change', 'Volume', 'Label', 'Forecast isnot a number because the prediction is in the future and there is no information on that data.
and 'i' is the forecast.
The list plus 1 value.

saving the classifier using Pickle.
save it after the data is trained, to save time. 