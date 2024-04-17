import pandas as pd
from catboost import CatBoostClassifier

data = pd.read_csv('bookings_per_year_month_day_season.csv')

# debug: display the first few rows of the DataFrame to valdate data integrity
# print(data.head())

# convert categorical variables to numerical format
data['month'] = pd.to_datetime(data['month'], format='%B').dt.month
data['season'] = data['season'].map({'Spring': 0, 'Summer': 1, 'Autumn': 2, 'Winter': 3})

# extract additional features from date columns
data['day_of_week'] = pd.to_datetime(data[['year', 'month', 'day']]).dt.dayofweek

# debug: display the updated DataFrame to verify changes
# print(data.head())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# split the data into training and testing sets
X = data[['year', 'month', 'day', 'season', 'day_of_week']]
y = data['bookings']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.impute import SimpleImputer

# handle missing values in dataset
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# linear regression
model = CatBoostClassifier(iterations=100)

# training time
model.fit(X_train_imputed, y_train)

# make predictions
y_pred = model.predict(X_test_imputed)

# evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

#
# how do I get a prediction for a certain day?
#
# year (literal int)
# month (literal int, January = 1...December = 12)
# day (also literal int, Jan 1st = 1...Jan 31 = 31)
# season (array, winter = 0...fall = 3)
# day (array, Monday = 0...Sunday = 6)
#
# The example below is supposed to
# gather a prediction for Monday, 
# Jan 1st, 2018
#

desired_date_features = pd.DataFrame({
    'year': [2018],
    'month': 1,
    'day': [1],
    'season': [0],
    'day_of_week': 0      
})
desired_date_features_imputed = imputer.transform(desired_date_features)
predicted_bookings = model.predict(desired_date_features_imputed)

print("Predicted number of bookings:", predicted_bookings[0])