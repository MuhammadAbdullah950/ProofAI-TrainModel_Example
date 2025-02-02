import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib  
import traceback  
import sys
from io import BytesIO  # Import BytesIO
import base64  # Import base64
import json  # Import json
import os

dataset_path = os.path.join(sys.argv[1], "iris.data.txt")
model_filename = sys.argv[2]
try:
    # Load the Dataset
    model_filename =model_filename + "iris.data.txt"
    column_names = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Species"]
    df = pd.read_csv(dataset_path, header=None, names=column_names)

    # Map species to numerical values
    df["Species"] = df["Species"].map({
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2
    })

    # Check for null values
    if df.isnull().sum().any():
        raise ValueError("Dataset contains missing values. Please clean the data and try again.")

    # Split Features and Labels
    X = df.drop(columns=["Species"])
    y = df["Species"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train KNN Model
    knn = KNeighborsClassifier(n_neighbors=3)  # k=3
    knn.fit(X_train, y_train)
    
    # Save model to BytesIO and encode to base64
    model_bytes = BytesIO()
    joblib.dump(knn, model_bytes)  # Write model into BytesIO buffer
    model_bytes.seek(0)  # Go to the beginning of the BytesIO buffer
    model_base64 = base64.b64encode(model_bytes.getvalue()).decode()  # Base64 encode the bytes

    response = {"model": model_base64, "error": None}
    print(json.dumps(response))

except Exception as e:
    # Catch exception and prepare error message
    error_message = str(e)
    response = {"model": None, "error": error_message}
    print(json.dumps(response))
