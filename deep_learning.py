"""Runnable deep-learning component: 3 hidden-layer scikit-learn neural network.

The deep candidate is trained/evaluated together with classical baselines inside
train_models.py, using the same training/validation/test partitions.
"""
from pathlib import Path
import joblib
from train_models import train


def load_deep_model():
    path = Path(__file__).resolve().parent / 'models' / 'deep_model.joblib'
    if not path.exists():
        train(model_dir=path.parent)
    return joblib.load(path)


if __name__ == '__main__':
    model = load_deep_model()
    print('Neural network hidden layers:', model.named_steps['model'].hidden_layer_sizes)
