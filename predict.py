
"""
Student Performance Predictor
Loads a trained machine-learning model and predicts student performance.

Expected model file:
    student_performance_model.pkl

The model must be trained and saved by train_model.py before predictions
can be made.
"""

from pathlib import Path
import argparse
import json

import joblib
import pandas as pd


# Project paths
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "student_performance_model.pkl"


def load_model(model_path=DEFAULT_MODEL_PATH):
    """Load and return the saved machine-learning model."""
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}\n"
            "Train the model first and save it using joblib.dump()."
        )

    return joblib.load(model_path)


def predict_student(student_data, model_path=DEFAULT_MODEL_PATH):
    """
    Predict a student's performance from a dictionary of input features.

    Args:
        student_data (dict): Student information used by the trained model.
        model_path (str or Path): Location of the saved model.

    Returns:
        dict: Predicted performance and, when available, confidence scores.
    """
    if not isinstance(student_data, dict) or not student_data:
        raise ValueError("student_data must be a non-empty dictionary.")

    model = load_model(model_path)

    # Use the feature names recorded during model training when available.
    expected_features = getattr(model, "feature_names_in_", None)

    if expected_features is not None:
        missing_features = [
            feature for feature in expected_features
            if feature not in student_data
        ]

        if missing_features:
            raise ValueError(
                "Missing required student features: "
                + ", ".join(missing_features)
            )

        input_data = {
            feature: student_data[feature]
            for feature in expected_features
        }
    else:
        # If feature names are unavailable, the input must match the
        # exact feature order used when training the model.
        input_data = student_data

    student_df = pd.DataFrame([input_data])

    prediction = model.predict(student_df)[0]

    result = {
        "predicted_performance": (
            prediction.item()
            if hasattr(prediction, "item")
            else prediction
        )
    }

    # Include class probabilities if the model supports them.
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(student_df)[0]
        classes = getattr(model, "classes_", range(len(probabilities)))

        result["class_probabilities"] = {
            str(label): round(float(probability), 4)
            for label, probability in zip(classes, probabilities)
        }

    return result


def main():
    """Allow predictions from the command line using JSON input."""
    parser = argparse.ArgumentParser(
        description="Predict student performance using a trained AI model."
    )
    parser.add_argument(
        "--input",
        required=True,
        help='Student features as JSON, e.g. \'{"attendance": 90}\'',
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the trained model file.",
    )

    args = parser.parse_args()

    try:
        student_data = json.loads(args.input)
        result = predict_student(student_data, args.model)
        print(json.dumps(result, indent=4, default=str))

    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Prediction error: {error}\n")


if __name__ == "__main__":
    main()
