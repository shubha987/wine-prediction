import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    data = pd.read_csv(file_path,delimiter=';')
    print("Data Loaded Successfully")
    print("Columns:", data.columns)
    return data

def preprocess_data(data):
    # Check if 'quality' column is present
    if 'quality' not in data.columns:
        raise KeyError("The 'quality' column is not present in the data")

    # Drop rows with missing values
    data.dropna(inplace=True)

    # Split data into features (X) and target (y)
    X = data.drop('quality', axis=1)
    y = data['quality']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
