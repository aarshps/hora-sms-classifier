import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

gas_booking_dates_df = pd.read_csv('gas_booking_dates.csv', parse_dates=['date'])
# Convert the date column to a datetime format
gas_booking_dates_df['date'] = pd.to_datetime(gas_booking_dates_df['date'])

# Calculate the time difference between each booking date and the previous booking date:
gas_booking_dates_df['time_diff'] = gas_booking_dates_df['date'].diff()

# Convert the time difference column to days:
gas_booking_dates_df['time_diff'] = gas_booking_dates_df['time_diff'].dt.days

# Impute the NaN values in the time difference column using the mean:
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
gas_booking_dates_df['time_diff'] = imputer.fit_transform(gas_booking_dates_df[['time_diff']])
X = gas_booking_dates_df[['time_diff']]
y = gas_booking_dates_df['date']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('Mean squared error:', np.mean((y_pred - y_test)**2))
# Get the current date
current_date = pd.to_datetime('2023-10-29')

# Calculate the time difference between the current date and the last booking date:
time_diff = current_date - gas_booking_dates_df['date'].iloc[-1]

# Convert the time difference to days:
time_diff = time_diff.dt.days

# Predict the next gas booking date:
next_gas_booking_date = model.predict([[time_diff]])[0]

print('Predicted next gas booking date:', next_gas_booking_date)
