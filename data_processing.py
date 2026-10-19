
import pandas as pd
from pathlib import Path


# Find the project folder and dataset
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "xAPI-Edu-Data.csv"


def load_data():
    """Load the student dataset from the CSV file."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}\n"
            "Please check that xAPI-Edu-Data.csv is inside the data folder."
        )

    data = pd.read_csv(DATA_PATH)

    if data.empty:
        raise ValueError("The student dataset is empty.")

    return data


def inspect_data(data):
    """Display information about the original dataset."""

    print("\n" + "=" * 50)
    print("STUDENT DATASET INFORMATION")
    print("=" * 50)

    print(f"\nNumber of student records: {data.shape[0]}")
    print(f"Number of columns: {data.shape[1]}")

    print("\nColumn names:")
    print(data.columns.tolist())

    print("\nFirst five records:")
    print(data.head().to_string(index=False))

    print("\nColumn data types:")
    print(data.dtypes)

    print("\nMissing values in each column:")
    print(data.isnull().sum())

    print(f"\nDuplicate records: {data.duplicated().sum()}")


def clean_data(data):
    """Clean the dataset without changing its original file."""

    cleaned_data = data.copy()

    # Remove duplicate records
    cleaned_data = cleaned_data.drop_duplicates()

    # Fill missing values in text columns with the most common value
    for column in cleaned_data.select_dtypes(
        include=["object", "category"]
    ).columns:
        if cleaned_data[column].isnull().any():
            most_common = cleaned_data[column].mode()

            if not most_common.empty:
                cleaned_data[column] = cleaned_data[column].fillna(
                    most_common.iloc[0]
                )

    # Fill missing numerical values with the column median
    for column in cleaned_data.select_dtypes(
        include=["number"]
    ).columns:
        if cleaned_data[column].isnull().any():
            median_value = cleaned_data[column].median()
            cleaned_data[column] = cleaned_data[column].fillna(
                median_value
            )

    return cleaned_data


def prepare_features(data):
    """Separate input information from the target column."""

    if "Class" not in data.columns:
        raise ValueError(
            "The dataset does not contain the expected 'Class' column."
        )

    # The AI will learn to predict the Class category
    X = data.drop(columns=["Class"])
    y = data["Class"]

    # Convert text categories into numerical columns
    X = pd.get_dummies(X, dtype=int)

    return X, y


def process_data():
    """Run the complete data-processing workflow."""

    # Step 1: Load the dataset
    data = load_data()

    # Step 2: Inspect the original data
    inspect_data(data)

    # Step 3: Clean the dataset
    cleaned_data = clean_data(data)

    print("\n" + "=" * 50)
    print("DATA CLEANING RESULTS")
    print("=" * 50)
    print(f"Records before cleaning: {len(data)}")
    print(f"Records after cleaning: {len(cleaned_data)}")
    print(f"Remaining missing values: {cleaned_data.isnull().sum().sum()}")

    # Step 4: Prepare the data for machine learning
    X, y = prepare_features(cleaned_data)

    print("\n" + "=" * 50)
    print("MACHINE LEARNING DATA PREPARATION")
    print("=" * 50)
    print(f"Number of input features: {X.shape[1]}")
    print(f"Number of target records: {len(y)}")

    print("\nTarget categories and their counts:")
    print(y.value_counts())

    print("\nPrepared input data (first five rows):")
    print(X.head().to_string(index=False))

    print("\nData processing completed successfully!")

    return X, y


if __name__ == "__main__":
    X, y = process_data()
