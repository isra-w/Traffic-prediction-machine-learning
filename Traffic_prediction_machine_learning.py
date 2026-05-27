
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import datetime

# Loading the data
df = pd.read_csv('traffic.csv')

# Basic preprocessing
print("Dataset shape:", df.shape)
print("\nUnique Traffic Situations:", df['Traffic Situation'].unique())

#feature engenering
# Convert Time to minutes
def time_to_minutes(t):
    try:
        time_obj = datetime.datetime.strptime(t.strip(), '%I:%M:%S %p')
        return time_obj.hour * 60 + time_obj.minute
    except:
        return np.nan

df['Time_Minutes'] = df['Time'].apply(time_to_minutes)

# Encode Day of the week
day_encoder = LabelEncoder()
df['Day_Encoded'] = day_encoder.fit_transform(df['Day of the week'])

# Features and target
features = ['Time_Minutes', 'Day_Encoded', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount']
X = df[features]
y = df['Traffic Situation']

# Handle any missing values
X = X.fillna(X.median())

# Split data: 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate on test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy on Test Set: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save encoders for future use
print("\nModel trained successfully!")

# Function to predict for user input
def predict_traffic(date_str, time_str):
    """
    Predict traffic situation for given date and time.
    date_str: e.g., '2025-05-15' or just day number like '15'
    time_str: e.g., '14:30:00 PM'
    """
    try:
        # Get day of week
        if len(date_str) == 2 or date_str.isdigit():  # Simple day number
            day_num = int(date_str)
            # Approximate day of week (Monday=0 ... Sunday=6)
            day_of_week = (day_num % 7)  # Rough approximation
        else:
            dt = datetime.datetime.strptime(date_str, '%Y-%m-%d')
            day_of_week = dt.weekday()
        
        # Time to minutes
        time_minutes = time_to_minutes(time_str)
        
        # Create input
        input_data = pd.DataFrame({
            'Time_Minutes': [time_minutes],
            'Day_Encoded': [day_of_week % 7],  # Rough mapping
            'CarCount': [0],      # We don't know counts - model will use time patterns
            'BikeCount': [0],
            'BusCount': [0],
            'TruckCount': [0]
        })
        
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        classes = model.classes_
        
        print(f"\nPrediction for {date_str} at {time_str}:")
        print(f"Traffic Situation: **{prediction}**")
        print("Confidence:")
        for cls, prob in zip(classes, probabilities):
            print(f"  {cls}: {prob*100:.1f}%")
            
    except Exception as e:
        print(f"Error making prediction: {e}")

# Interactive prediction loop
if __name__ == "__main__":
    print("\n=== Traffic Prediction System ===")
    while True:
        print("\nEnter date (e.g., '15' or '2025-05-15') and time (e.g., '14:30:00 PM')")
        date_input = input("Date: ").strip()
        if date_input.lower() == 'exit':
            break
        time_input = input("Time: ").strip()
        
        predict_traffic(date_input, time_input)