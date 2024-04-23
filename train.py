import pandas as pd
from catboost import CatBoostRegressor

data = pd.read_csv('bookings_per_year_month_day_season_adults.csv')

# convert categorical variables to numerical format
data['month'] = pd.to_datetime(data['month'], format='%B').dt.month
data['season'] = data['season'].map({'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3})

# extract additional features from date columns
data['day_of_week'] = pd.to_datetime(data[['year', 'month', 'day']]).dt.dayofweek

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#print(data.head())

# split the data into training and testing sets
X = data[['year', 'month', 'day', 'season', 'adults', 'day_of_week']]
y = data['bookings']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# linear regression
model = CatBoostRegressor(n_estimators=100, learning_rate=0.1, depth=6, verbose=0, random_seed=5, l2_leaf_reg=9)

# training time
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)

# evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)
print("R-squared:", r2)

#
# how do I get a prediction for a certain day?
#
# year (literal int)
# month (literal int, January = 1...December = 12)
# day (also literal int, Jan 1st = 1...Jan 31 = 31)
# season (array, winter = 0...fall = 3)
# day (array, Monday = 1...Sunday = 7)
#
# The example below is supposed to
# gather a prediction for Monday, 
# Jan 1st, 2018
#

desired_date_features = pd.DataFrame({
    'year': [2018],
    'month': [1],
    'day': [1],
    'season': [3],
    'adults': [2],
    'day_of_week': [1]
})
predicted_bookings = model.predict(desired_date_features)

print("Predicted number of bookings:", predicted_bookings[0])