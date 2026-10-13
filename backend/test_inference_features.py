from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd

from app.features.inference_features import (
    add_inference_row,
    add_inference_player_status,
)

from app.features.player_features import (
    add_receiving_features,
    add_rb_features,
    add_qb_features,
    add_kicker_features,
)

from app.features.matchup_features import add_matchup_features

from app.models.predictor import predict_player, FEATURE_CONFIG


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BACKEND_DIR = Path(__file__).parent
PROJECT_ROOT = BACKEND_DIR.parent


# ---------------------------------------------------------
# Load saved feature dataset
# ---------------------------------------------------------

saved_df = pd.read_parquet(
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player_features.parquet"
)


# ---------------------------------------------------------
# Load raw historical player stats
# ---------------------------------------------------------

player_stats = nfl.load_player_stats(
    range(2018, 2026)
)

df = player_stats.to_pandas()

df = df[
    df["season_type"] == "REG"
].copy()


# ---------------------------------------------------------
# Load player metadata
# ---------------------------------------------------------

players = nfl.load_players().to_pandas()

player_metadata = (
    players[
        [
            "gsis_id",
            "rookie_season",
        ]
    ]
    .copy()
    .rename(
        columns={
            "gsis_id": "player_id"
        }
    )
)

df = df.merge(
    player_metadata,
    on="player_id",
    how="left",
)


# ---------------------------------------------------------
# Build player season status
# ---------------------------------------------------------

player_seasons = (
    df[
        [
            "player_id",
            "season",
        ]
    ]
    .dropna(
        subset=["player_id"]
    )
    .drop_duplicates()
    .sort_values(
        [
            "player_id",
            "season",
        ]
    )
)

player_seasons["last_active_season"] = (
    player_seasons
    .groupby("player_id")["season"]
    .shift(1)
)

player_seasons["season_gap"] = (
    player_seasons["season"]
    - player_seasons["last_active_season"]
)

df = df.merge(
    player_seasons,
    on=[
        "player_id",
        "season",
    ],
    how="left",
)

df["is_rookie"] = (
    df["season"]
    == df["rookie_season"]
).astype(int)

df["is_veteran_after_gap"] = (
    (df["season"] > df["rookie_season"])
    &
    (
        df["last_active_season"].isna()
        | (df["season_gap"] > 1)
    )
).astype(int)

df = df.sort_values(
    [
        "player_id",
        "season",
        "week",
    ]
).reset_index(drop=True)


# ---------------------------------------------------------
# Features to validate for each position
# ---------------------------------------------------------

POSITION_FEATURES = {
    "WR": [
        "targets_avg_3",
        "targets_avg_5",
        "receiving_yards_avg_3",
        "receiving_yards_avg_5",
        "target_share_avg_3",
        "target_share_avg_5",
        "targets_trend",
        "rec_yards_trend",
    ],

    "TE": [
        "targets_avg_3",
        "targets_avg_5",
        "receiving_yards_avg_3",
        "receiving_yards_avg_5",
        "target_share_avg_3",
        "target_share_avg_5",
        "targets_trend",
        "rec_yards_trend",
    ],

    "RB": [
        "targets_avg_3",
        "receiving_yards_avg_3",
        "rb_carries_avg_3",
        "rb_carries_avg_5",
        "rb_rushing_yards_avg_3",
        "rb_rushing_yards_avg_5",
        "rb_opportunities_avg_3",
        "rb_opportunities_avg_5",
        "rb_carries_trend",
        "rb_rushing_yards_trend",
        "rb_opportunities_trend",
    ],

    "QB": [
        "attempts_avg_3",
        "attempts_avg_5",
        "passing_yards_avg_3",
        "passing_yards_avg_5",
        "passing_tds_avg_3",
        "passing_interceptions_avg_3",
        "passing_air_yards_avg_3",
        "qb_carries_avg_3",
        "qb_rushing_yards_avg_3",
        "attempts_trend",
        "passing_yards_trend",
        "qb_rushing_yards_trend",
    ],

    "K": [
        "fg_att_avg_3",
        "fg_att_avg_5",
        "fg_made_avg_3",
        "fg_made_avg_5",
        "fg_long_avg_3",
        "pat_att_avg_3",
        "pat_made_avg_3",
        "fg_att_trend",
        "fg_made_trend",
    ],
}

MATCHUP_FEATURES = [
    "opp_points_allowed_avg_3",
    "opp_points_allowed_avg_5",
    "opp_points_allowed_trend"
]


# ---------------------------------------------------------
# Run the historical simulation for one position
# ---------------------------------------------------------

def test_position(position, season=2025, week=10):

    print(
        "\n"
        + "=" * 60
    )

    print(
        f"Testing {position} | "
        f"{season} Week {week}"
    )

    print(
        "=" * 60
    )

    # Select one real historical player-game
    test_row = saved_df[
        (saved_df["season"] == season)
        & (saved_df["week"] == week)
        & (saved_df["position"] == position)
    ].iloc[0]

    player_id = test_row["player_id"]
    team = test_row["team"]
    opponent_team = test_row["opponent_team"]
    rookie_season = test_row["rookie_season"]

    print(
        f"Player: {player_id}"
    )

    # -----------------------------------------------------
    # Simulate the dataset before the target week
    # -----------------------------------------------------

    history_df = df[
        (df["season"] < season)
        |
        (
            (df["season"] == season)
            & (df["week"] < week)
        )
    ].copy()

    # -----------------------------------------------------
    # Add synthetic upcoming-game row
    # -----------------------------------------------------

    history_df = add_inference_row(
        history_df,
        player_id=player_id,
        position=position,
        team=team,
        opponent_team=opponent_team,
        season=season,
        week=week,
    )

    # -----------------------------------------------------
    # Add synthetic player status
    # -----------------------------------------------------

    history_df = add_inference_player_status(
        history_df,
        player_id=player_id,
        season=season,
        week=week,
        rookie_season=rookie_season,
    )

    # -----------------------------------------------------
    # Run the appropriate player feature pipelines
    # -----------------------------------------------------

    if position in [
        "WR",
        "TE",
        "RB",
    ]:
        history_df = add_receiving_features(
            history_df
        )

    if position == "RB":
        history_df = add_rb_features(
            history_df
        )

    if position == "QB":
        history_df = add_qb_features(
            history_df
        )

    if position == "K":
        history_df = add_kicker_features(
            history_df
        )
    
    history_df = add_matchup_features(
        history_df
    )

    # -----------------------------------------------------
    # Extract the engineered synthetic row
    # -----------------------------------------------------

    synthetic_row = history_df[
        (history_df["player_id"] == player_id)
        & (history_df["season"] == season)
        & (history_df["week"] == week)
    ].iloc[0]

    required_features = FEATURE_CONFIG[position]["features"]

    player_features = {
        feature: synthetic_row[feature]
        for feature in required_features
    }

    prediction = predict_player(
        position,
        player_features,
    )

    print(
        f"\nProjected fantasy points: "
        f"{prediction:.2f}"
    )

    # -----------------------------------------------------
    # Compare against the original saved historical row
    # -----------------------------------------------------
    
    all_match = True

    features_to_test = (
        POSITION_FEATURES[position]
        + MATCHUP_FEATURES
    )

    for feature in features_to_test:

        generated = synthetic_row[feature]

        saved = test_row[
            feature
        ]

        matches = np.isclose(
            generated,
            saved,
            equal_nan=True,
        )

        status = (
            "PASS"
            if matches
            else "FAIL"
        )

        print(
            f"{feature}: "
            f"{status} "
            f"(generated={generated}, "
            f"saved={saved})"
        )

        if not matches:
            all_match = False

    print(
        f"\n{position} synthetic features match:",
        all_match,
    )

    return all_match


# ---------------------------------------------------------
# Test all supported fantasy positions
# ---------------------------------------------------------

results = {}

for position in [
    "WR",
    "TE",
    "RB",
    "QB",
    "K",
]:
    results[position] = test_position(
        position
    )


# ---------------------------------------------------------
# Final summary
# ---------------------------------------------------------

print(
    "\n"
    + "=" * 60
)

print(
    "FINAL RESULTS"
)

print(
    "=" * 60
)

for position, passed in results.items():

    print(
        f"{position}: "
        f"{'PASS' if passed else 'FAIL'}"
    )


print(
    "\nAll player inference tests passed:",
    all(
        results.values()
    ),
)