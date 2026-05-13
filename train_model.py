import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── 1. Load Data ──
df = pd.read_csv('Data_Train.csv')
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# ── 2. Feature Engineering ──

# Journey date components
df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'], format="%d/%m/%Y")
df['date']  = df['Date_of_Journey'].dt.day
df['month'] = df['Date_of_Journey'].dt.month
df['year']  = df['Date_of_Journey'].dt.year

# Departure time
df['Dep_Time']    = pd.to_datetime(df['Dep_Time'])
df['dep_hour']    = df['Dep_Time'].dt.hour
df['dep_minutes'] = df['Dep_Time'].dt.minute

# Arrival time
df['Arrival_Time']    = pd.to_datetime(df['Arrival_Time'])
df['arr_hour']        = df['Arrival_Time'].dt.hour
df['arr_minutes']     = df['Arrival_Time'].dt.minute

# Duration in hours and minutes
df['hours']   = df['Duration'].str.extract(r'(\d+)h').fillna(0).astype(int)
df['minutes'] = df['Duration'].str.extract(r'(\d+)m').fillna(0).astype(int)
df['Duration_total_mins'] = df['hours'] * 60 + df['minutes']

# Total stops and airports from Route
df['Total_Stops']          = df['Route'].apply(lambda x: len(x.split('→')) - 1 if '→' in x else len(x.split('?')) - 1)
df['Source_airport']       = df['Route'].apply(lambda x: x.split('→')[0].strip() if '→' in x else x.split('?')[0].strip())
df['Destination_airport']  = df['Route'].apply(lambda x: x.split('→')[-1].strip() if '→' in x else x.split('?')[-1].strip())

# ── 3. Label Encoding ──
le_airline             = LabelEncoder()
le_source              = LabelEncoder()
le_destination         = LabelEncoder()
le_additional          = LabelEncoder()
le_source_airport      = LabelEncoder()
le_destination_airport = LabelEncoder()

df['Airline']             = le_airline.fit_transform(df['Airline'])
df['Source']              = le_source.fit_transform(df['Source'])
df['Destination']         = le_destination.fit_transform(df['Destination'])
df['Additional_Info']     = le_additional.fit_transform(df['Additional_Info'])
df['Source_airport']      = le_source_airport.fit_transform(df['Source_airport'])
df['Destination_airport'] = le_destination_airport.fit_transform(df['Destination_airport'])

# ── 4. Features and Target ──
X = df[['Airline', 'Source', 'Destination', 'Total_Stops', 'Additional_Info',
        'date', 'month', 'year',
        'dep_hour', 'dep_minutes',
        'arr_hour', 'arr_minutes',
        'hours', 'minutes', 'Duration_total_mins',
        'Source_airport', 'Destination_airport']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── 5. Train XGBoost ──
print("Training XGBoost model...")
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42
)
model.fit(X_train, y_train)

# ── 6. Evaluate ──
y_pred = model.predict(X_test)
print(f"MAE:  {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²:   {r2_score(y_test, y_pred):.4f}")

# ── 7. Save model and all encoders ──
with open('model.pkl',                    'wb') as f: pickle.dump(model,                  f)
with open('le_airline.pkl',               'wb') as f: pickle.dump(le_airline,              f)
with open('le_source.pkl',                'wb') as f: pickle.dump(le_source,               f)
with open('le_destination.pkl',           'wb') as f: pickle.dump(le_destination,          f)
with open('le_additional.pkl',            'wb') as f: pickle.dump(le_additional,           f)
with open('le_source_airport.pkl',        'wb') as f: pickle.dump(le_source_airport,       f)
with open('le_destination_airport.pkl',   'wb') as f: pickle.dump(le_destination_airport,  f)

print("Done! All .pkl files saved successfully.")
print("You can now run: python app.py")
