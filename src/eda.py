# src/eda.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def perform_eda(data_path):
    # Load the data
    data = pd.read_csv(data_path, sep=';')
    
    # Basic statistics and info
    print("Data Information:")
    print(data.info())
    print("\nData Description:")
    print(data.describe())

    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

    # Distribution of wine quality
    plt.figure(figsize=(10, 6))
    sns.countplot(x='quality', data=data)
    plt.title('Distribution of Wine Quality')
    plt.xlabel('Quality')
    plt.ylabel('Count')
    plt.show()

    # Pairplot of features
    sns.pairplot(data, hue='quality')
    plt.title('Pairplot of Features')
    plt.show()

# Example usage:
# perform_eda('../data/winequality-red.csv')
