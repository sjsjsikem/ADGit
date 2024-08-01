# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import networkx as nx
import clubear as cb
import warnings
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore') # 在整个Python脚本执行期间忽略所有的警告。

'''
Step 1: Load and Combine Data
Adjust the selected_years list in to include the years the user wishes to analyse.
'''
def load_flight_data(years):
    data_frames = []
    for year in years:
        file_path = f"{year}.csv"
        if os.path.exists(file_path):
            data_frames.append(pd.read_csv(file_path))
        else:
            print(f"Warning: File for year {year} not found.")
    combined_data = pd.concat(data_frames, ignore_index=True)
    return combined_data

# Choose the years you want to load.
#这里是初步的加载程序，加载全部年份的时候就从这里加载
selected_years = [2004, 2005]  
# It can be replaced with the desired years.
#翻译： 它可以被替换为所需的年份。

# Load the data for the selected years.
combined_flight_data = load_flight_data(selected_years)

# Basic data exploration
print(combined_flight_data.shape)
combined_flight_data.info()


'''
Step 2: Clean Data
'''
# Remove duplicate rows
combined_flight_data.drop_duplicates(inplace=True)

# Handle missing values
cleaned_combined_flight_data = combined_flight_data.fillna(0)

# Replace airline codes with full names
airline_names = {
    'UA': 'United Airlines', 'US': 'United States Airways', 'WN': 'Southwest Airlines',
    'NW': 'Northwest Airlines', 'OH': 'PSA Airlines', 'OO': 'SkyWest Airlines',
    'XE': 'Expressjet Airlines', 'TZ': 'Air Tazania Airlines', 'DL': 'Delta Airlines',
    'EV': 'Atlantic Southeast Airlines', 'FL': 'Florida Airlines', 'HA': 'Hawaiian Airlines',
    'HP': 'America West Airlines', 'MQ': 'Envoy Airlines', 'AA': 'American Airlines',
    'AS': 'Alaska Airlines', 'B6': 'JetBlue Airways', 'CO': 'Continental Airlines',
    'DH': 'Indepedence Airlines', 'F9': 'Frontier Airlines'
}
cleaned_combined_flight_data['UniqueCarrier'].replace(airline_names, inplace=True)

# Export cleaned data to CSV
cleaned_combined_flight_data.to_csv('cleaned_combined_flight_data.csv', index=False)


'''
Step 3: Feature Engineering
'''
# Create new features
cleaned_combined_flight_data['DepTime'] = cleaned_combined_flight_data['DepTime'].astype(float)
bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400]
labels = ["00:00", "01:00", "02:00", "03:00", "04:00", "05:00", "06:00", "07:00", "08:00", "09:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", "17:00", "18:00", "19:00", "20:00", "21:00", "22:00", "23:00", "24:00"]
cleaned_combined_flight_data['24HoursTime'] = pd.cut(cleaned_combined_flight_data['DepTime'], bins=bins, labels=labels, include_lowest=True)


'''
Step 4: Train Machine Learning Models
'''
# Prepare data for training
features = ['Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'CRSDepTime', 'ArrTime', 'CRSArrTime', 'UniqueCarrier', 'FlightNum', 'TailNum', 'ActualElapsedTime', 'CRSElapsedTime', 'AirTime', 'ArrDelay', 'DepDelay', 'Origin', 'Dest', 'Distance', 'TaxiIn', 'TaxiOut', 'Cancelled', 'CancellationCode', 'Diverted', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']
target = 'DepDelay'

# Convert categorical features to numerical values
cleaned_combined_flight_data = pd.get_dummies(cleaned_combined_flight_data, columns=['UniqueCarrier', 'Origin', 'Dest'])

X = cleaned_combined_flight_data[features]
y = cleaned_combined_flight_data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Save the model to a pkl file for server deployment
with open('flight_delay_model.pkl', 'wb') as file:
    pickle.dump(model, file)


'''
Step 5: Analysis and Visualisation
'''
# Best times to fly to minimise delays
best_time_of_day = cleaned_combined_flight_data.groupby("24HoursTime")["DepDelay"].sum()

# Plot the variation of delay time at different times of the day
plt.figure(figsize=(15, 9))
plt.plot(best_time_of_day, marker="o")
plt.title("Variation of Delay Time at Different Times of the Day", fontsize=18)
plt.xlabel("Time of the Day", fontsize=15)
plt.ylabel("Total Delay Minutes", fontsize=15)
plt.ticklabel_format(style="plain", axis="y")
plt.show()

# Best day of the week to avoid delays
best_day_of_week = cleaned_combined_flight_data.groupby("DayOfWeek")["DepDelay"].sum()

# Plot the variation of delay time for different days in a week
plt.figure(figsize=(15, 8))
plt.plot(best_day_of_week, color="green", marker="o")
plt.title("Variation of Delay Time for Different Days in a Week", fontsize=18)
plt.xlabel("Day of the Week", fontsize=15)
plt.ylabel("Total Delay Minutes", fontsize=15)
plt.ticklabel_format(style="plain", axis="y")
days = [1, 2, 3, 4, 5, 6, 7]
days_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
ax = plt.subplot()
ax.set_xticks(days)
ax.set_xticklabels(days_name, fontsize=13)
plt.show()

# Best day of the month to avoid delays
best_day_of_month = cleaned_combined_flight_data.groupby("DayofMonth")["DepDelay"].sum()

# Plot the variation of delay time for different days of the month
plt.figure(figsize=(15, 9))
plt.plot(best_day_of_month, color="green", marker="o")
plt.title("Variation of Delay Time for Different Days of the Month", fontsize=18)
plt.xlabel("Day of the Month", fontsize=15)
plt.ylabel("Total Delay Minutes", fontsize=15)
plt.ticklabel_format(style="plain", axis="y")
days_of_month = range(1, 32)
ax = plt.subplot()
ax.set_xticks(days_of_month)
plt.show()

# Best month of the year to avoid delays
best_month_of_year = cleaned_combined_flight_data.groupby("Month")["DepDelay"].sum()

# Plot the variation of delay time for different months in a year
plt.figure(figsize=(15, 8))
plt.plot(best_month_of_year, color="green", marker="o")
plt.title("Variation of Delay Time for Different Months in a Year", fontsize=18)
plt.xlabel("Months of the Year", fontsize=15)
plt.ylabel("Total Delay Minutes", fontsize=15)
plt.ticklabel_format(style="plain", axis="y")
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
months = range(1, 13)
ax = plt.subplot()
ax.set_xticks(months)
ax.set_xticklabels(month_names, fontsize=13)
plt.show()


'''
Step 6: Correlation Analysis
'''
# Correlation between aircraft age and delays
airlines = pd.concat(map(pd.read_csv, ['2004.csv', '2005.csv']))
airports = pd.read_csv('airports.csv')
planes = pd.read_csv('plane-data.csv')

# Clean and preprocess data
airlines = airlines.dropna(subset=['TailNum'])
planes = planes.dropna(subset=['year'])
planes = planes.rename(columns={'year': 'ManufactureYear'})
airlines = airlines.merge(planes[['tailnum', 'ManufactureYear']], left_on='TailNum', right_on='tailnum', how='left')

# Create delay indicator
airlines['ADelay'] = np.where(airlines['ArrDelay'] > 0, 1, 0)

# Group by manufacture year and delay
df_planes_grouped = airlines.groupby(['ManufactureYear', 'ADelay']).size().reset_index(name='Counts')
df_planes_grouped['TotalFlights'] = airlines.groupby('ManufactureYear').size().values
df_planes_grouped['DelayPercentage'] = (df_planes_grouped['Counts'] / df_planes_grouped['TotalFlights']) * 100

# Plot delay percentage by manufacture year
df_planes_grouped = df_planes_grouped[df_planes_grouped['ADelay'] == 1]
plt.figure(figsize=(15, 8))
plt.plot(df_planes_grouped['ManufactureYear'], df_planes_grouped['DelayPercentage'], color='green', marker='o')
plt.title('Percentage of Delays by Aircraft Age', fontsize=18)
plt.xlabel('Year of Manufacture', fontsize=15)
plt.ylabel('Delay Percentage', fontsize=15)
plt.show()


'''
Step 7: Integrate with Flask Application
'''
from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('flight_delay_model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict(pd.DataFrame([data]))
    return jsonify(prediction=prediction[0])

if __name__ == '__main__':
    app.run(port=5000, debug=True)
