"""
Student Performance Predictor
Loads the model trained by train_model.py and predicts
whether a student's performance is Low, Medium, or High.
"""

from pathlib import Path
import argparse
import json

import joblib
import pandas as pd


# Project paths
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = (
    BASE_DIR / "models" / "student_performance_model.pkl"
)

# Meaning of the labels in the xAPI-Edu-Data dataset
PERFORMANCE_LABELS = {
    "L": "Low",
    "M": "Medium",
    "H": "High",
}


def load_model(model_path=DEFAULT_MODEL_PATH):
    """Load the trained model from the models folder."""
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at: {model_path}\n"
            "Run train_model.py first to train and save the model."
        )

    return joblib.load(model_path)


def predict_student(student_data, model_path=DEFAULT_MODEL_PATH):
    """
    Predict student performance.

    student_data must contain the original dataset feature names,
    excluding the target column 'Class'.
    """
    if not isinstance(student_data, dict) or not student_data:
        raise ValueError(
            "Student data must be a non-empty JSON object."
        )

    if "Class" in student_data:
        raise ValueError(
            "Do not provide 'Class'. It is the value being predicted."
        )

    model = load_model(model_path)

    # The training pipeline expects the original dataset columns.
    expected_features = getattr(model, "feature_names_in_", None)

    if expected_features is None:
        raise ValueError(
            "The saved model does not contain feature names. "
            "Please retrain it using the supplied train_model.py."
        )

    missing_features = [
        feature for feature in expected_features
        if feature not in student_data
    ]

    if missing_features:
        raise ValueError(
            "Missing required features: "
            + ", ".join(missing_features)
        )

    # Keep the same feature order used during training.
    input_data = {
        feature: student_data[feature]
        for feature in expected_features
    }

    student_df = pd.DataFrame([input_data])

    # Predict the student's performance label.
    prediction = model.predict(student_df)[0]
    label = str(prediction)

    result = {
        "predicted_label": label,
        "predicted_performance": PERFORMANCE_LABELS.get(
            label, "Unknown"
        ),
    }

    # Include probabilities if supported by the trained model.
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(student_df)[0]

        # The pipeline exposes the class labels from its classifier.
        classes = model.classes_

        result["class_probabilities"] = {
            PERFORMANCE_LABELS.get(str(class_label), str(class_label)):
                round(float(probability), 4)
            for class_label, probability
            in zip(classes, probabilities)
        }

    return result


def main():
    """Run predictions from the command line."""
    parser = argparse.ArgumentParser(
        description="Predict student performance using the trained model."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="JSON text or path to a JSON input file.",
    )

    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Optional path to the saved model.",
    )

    args = parser.parse_args()

    try:
        input_path = Path(args.input)

        if input_path.is_file():
            student_data = json.loads(
                input_path.read_text(encoding="utf-8")
            )
        else:
            student_data = json.loads(args.input)

        result = predict_student(
            student_data,
            model_path=args.model,
        )

        print(json.dumps(result, indent=4))

    except (ValueError, FileNotFoundError, json.JSONDecodeError) as error:
        parser.exit(1, f"Prediction error: {error}\n")


if __name__ == "__main__":
    main()