import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "data/xAPI-Edu-Data.csv"
TARGET = "Class"                      # L = low, M = medium, H = high
LABELS = {"L": 0, "M": 1, "H": 2}


def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def clean_data(df):
    df = df.copy()

    # 1. Fix inconsistent column names
    df = df.rename(columns={
        "NationalITy": "Nationality",
        "PlaceofBirth": "PlaceOfBirth",
        "VisITedResources": "VisitedResources",
        "raisedhands": "RaisedHands",
        "ParentschoolSatisfaction": "ParentSchoolSatisfaction",
    })

    # 2. Remove duplicates and report missing values
    df = df.drop_duplicates()
    print("Missing values:", int(df.isna().sum().sum()))
    df = df.dropna()

    # 3. Standardise misspelled text values
    fixes = {"KuwaIT": "Kuwait", "KW": "Kuwait", "lebanon": "Lebanon",
             "Tunis": "Tunisia", "Lybia": "Libya", "venzuela": "Venezuela",
             "SaudiArabia": "Saudi Arabia", "USA": "United States"}
    for col in ["Nationality", "PlaceOfBirth"]:
        df[col] = df[col].replace(fixes).str.strip()

    return df


def encode_features(df):
    """Turn text columns into numbers the models can use."""
    y = df[TARGET].map(LABELS)
    X = df.drop(columns=[TARGET])
    X = pd.get_dummies(X, drop_first=True)   # one-hot encoding
    return X, y


def get_train_test(test_size=0.2, seed=42):
    """Full pipeline: returns X_train, X_test, y_train, y_test."""
    df = clean_data(load_data())
    X, y = encode_features(df)
    return train_test_split(X, y, test_size=test_size,
                            random_state=seed, stratify=y)


if __name__ == "__main__":
    df = clean_data(load_data())
    print("Shape after cleaning:", df.shape)
    print(df[TARGET].value_counts())
    X_train, X_test, y_train, y_test = get_train_test()
    print("Features:", X_train.shape[1])
    print("Train rows:", len(X_train), "| Test rows:", len(X_test))
    df.to_csv("cleaned_student_data.csv", index=False)
    print("Saved cleaned_student_data.csv")
