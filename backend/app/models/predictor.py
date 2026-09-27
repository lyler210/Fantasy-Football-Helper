from pathlib import Path
import pandas as pd
import json
import joblib

ARTIFACT_DIR = Path(__file__).parent / "artifacts"

MODEL_FILES = {
    "WR": "wr_model.joblib",
    "TE": "te_model.joblib",
    "RB": "rb_model.joblib",
    "QB": "qb_model.joblib",
    "K": "k_model.joblib",
}


def load_models():
    models = {}

    for position, filename in MODEL_FILES.items():
        model_path = ARTIFACT_DIR / filename
        models[position] = joblib.load(model_path)

    return models

def load_feature_config():
    config_path = ARTIFACT_DIR / "feature_config.json"

    with open(config_path, "r") as file:
        return json.load(file)

MODELS = load_models()
FEATURE_CONFIG = load_feature_config()

def predict_player(position, player_features):
    position = position.upper()

    if position not in MODELS:
        raise ValueError(
            f"Unsupported position: {position}. "
            f"Expected one of {list(MODELS.keys())}."
        )
    
    required_features = FEATURE_CONFIG[position]["features"]

    missing_features = [
        feature
        for feature in required_features
        if feature not in player_features
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features for {position}: "
            f"{missing_features}"
        )
    
    ordered_features = {
        feature: player_features[feature]
        for feature in required_features
    }

    input_df = pd.DataFrame(
        [ordered_features]
    )

    model = MODELS[position]

    prediction = model.predict(input_df)[0]

    return float(prediction)