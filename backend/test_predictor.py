from pathlib import Path
import pandas as pd

from app.models.predictor import predict_player


BACKEND_DIR = Path(__file__).parent
PROJECT_ROOT = BACKEND_DIR.parent

df = pd.read_parquet(
    PROJECT_ROOT / "data" / "processed" / "player_features.parquet"
)

positions = ["WR", "TE", "RB", "QB", "K"]

for position in positions:
    test_player = df[
        (df["position"] == position) &
        (df["season"] == 2025)
    ].iloc[0]

    player_features = test_player.to_dict()

    prediction = predict_player(
        position,
        player_features
    )

    if position == "K":
        actual = test_player["kicker_fantasy_points"]
    else:
        actual = test_player["fantasy_points_ppr"]

    print()
    print("Position:", position)
    print("Player:", test_player["player_display_name"])
    print("Week:", test_player["week"])
    print("Prediction:", prediction)
    print("Actual:", actual)