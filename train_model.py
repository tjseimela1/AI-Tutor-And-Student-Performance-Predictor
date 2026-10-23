import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Get the main project folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set the file locations
DATA_PATH = os.path.join(
    BASE_DIR, "data", "xAPI-Edu-Data.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(
    MODEL_DIR, "student_performance_model.pkl"
)


def train_model():
    # Check if the dataset exists
    if not os.path.exists(DATA_PATH):
        print("Error: Student dataset not found.")
        print("Expected location:", DATA_PATH)
        return

    # Load the student dataset
    data = pd.read_csv(DATA_PATH)

    # Remove rows with missing values
    data = data.dropna()

    # Select the column we want to predict
    target_column = "Class"

    if target_column not in data.columns:
        print("Error: The dataset must contain a 'Class' column.")
        return

    # Separate student information from performance labels
    X = data.drop(columns=[target_column])
    y = data[target_column]

    if y.nunique() < 2:
        print("Error: The dataset needs at least two classes.")
        return

    # Identify text and numerical columns
    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # Convert text information into numbers
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns
            ),
            (
                "numerical",
                "passthrough",
                numerical_columns
            )
        ]
    )

    # Create the AI model
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                )
            )
        ]
    )

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Train the model
    print("Training the student performance model...")
    model.fit(X_train, y_train)

    # Test the model
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nTraining completed!")
    print(f"Test accuracy: {accuracy:.2%}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # Create the models folder if needed
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save the trained model
    joblib.dump(model, MODEL_PATH)

    print("\nModel saved successfully!")
    print("Saved at:", MODEL_PATH)


if __name__ == "__main__":
    train_model()