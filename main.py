from src.data_preprocessing import load_data, preprocess_data
from src.model_training import train_model, evaluate_model

# Load the data
data_file_path = 'data/winequality-red.csv'
data = load_data(data_file_path)

# Print the first few rows of the data for verification
print(data.head())

# Preprocess the data
X_train, X_test, y_train, y_test = preprocess_data(data)

# Train the model
model = train_model(X_train, y_train)

# Evaluate the model
accuracy = evaluate_model(model, X_test, y_test)
print(f'Model Accuracy: {accuracy * 100:.2f}%')
