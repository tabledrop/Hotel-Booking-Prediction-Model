import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Load data
data = pd.read_csv('bookings_per_year_month_day_season.csv')

# Convert categorical variables to numerical format
data['month'] = pd.to_datetime(data['month'], format='%B').dt.month
data['season'] = data['season'].map({'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3})

# Extract additional features from date columns
data['day_of_week'] = pd.to_datetime(data[['year', 'month', 'day']]).dt.dayofweek

# Split the data into training and testing sets
X = data[['year', 'month', 'day', 'season', 'day_of_week']]
y = data['bookings']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Convert data
X_train_tensor = torch.tensor(X_train_imputed, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_imputed, dtype=torch.float32)

# Make a class for the model
class BookingsPredictor(nn.Module):
    def __init__(self, input_size):
        super(BookingsPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the model
model = BookingsPredictor(input_size=X_train.shape[1])

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

# Make predictions
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    y_pred = y_pred_tensor.numpy()

# Evaluate model & print out values
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

'''
How do I get a prediction for a certain day?
Year (literal int)
Month (literal int, January = 1...December = 12)
Day (also literal int, Jan 1st = 1...Jan 31 = 31)
Season (array, Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3)
Day (array, Monday = 0...Sunday = 6)
The example below is supposed to
gather a prediction for Monday,Jan 1st, 2018
'''

desired_date_features = pd.DataFrame({
    'year': [2018],
    'month': [1],
    'day': [1],
    'season': [3],
    'day_of_week': [0]
})

desired_date_features_imputed = imputer.transform(desired_date_features)
desired_date_tensor = torch.tensor(desired_date_features_imputed, dtype=torch.float32)
predicted_bookings_tensor = model(desired_date_tensor)
rounded_predicted_bookings = round(predicted_bookings_tensor.item())

print("Predicted number of bookings:", rounded_predicted_bookings)
