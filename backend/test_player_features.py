from pathlib import Path

import nflreadpy as nfl
import numpy as np
import pandas as pd

from app.features.player_features import (
    add_receiving_features,
    add_rb_features,
    add_qb_features,
    add_kicker_features,
)
from app.features.matchup_features import add_matchup_features

# Same raw input used during feature engineering
player_stats = nfl.load_player_stats(range(2018, 2026))
raw_df = player_stats.to_pandas()

raw_df = raw_df[
    raw_df["season_type"] == "REG"
].copy()


# Player metadata
players = nfl.load_players().to_pandas()

player_metadata = players[
    ["gsis_id", "rookie_season"]
].copy()

player_metadata = player_metadata.rename(
    columns={"gsis_id": "player_id"}
)

raw_df = raw_df.merge(
    player_metadata,
    on="player_id",
    how="left"
)

player_seasons = (
    raw_df[
        ["player_id", "season"]
    ]
    .dropna(subset=["player_id"])
    .drop_duplicates()
    .sort_values(["player_id", "season"])
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

raw_df = raw_df.merge(
    player_seasons,
    on=["player_id", "season"],
    how="left"
)

raw_df["is_rookie"] = (
    raw_df["season"] == raw_df["rookie_season"]
).astype(int)

raw_df["is_veteran_after_gap"] = (
    (raw_df["season"] > raw_df["rookie_season"])
    &
    (
        raw_df["last_active_season"].isna()
        |
        (raw_df["season_gap"] > 1)
    )
).astype(int)

raw_df = raw_df.sort_values(
    ["player_id", "season", "week"]
).reset_index(drop=True)

raw_df = add_receiving_features(raw_df)
raw_df = add_rb_features(raw_df)
raw_df = add_qb_features(raw_df)
raw_df = add_kicker_features(raw_df)
raw_df = add_matchup_features(raw_df)

raw_df = raw_df[
    raw_df["season"] >= 2019
].copy()


BACKEND_DIR = Path(__file__).parent
PROJECT_ROOT = BACKEND_DIR.parent

saved_df = pd.read_parquet(
    PROJECT_ROOT / "data" / "processed" / "player_features.parquet"
)

receiving_stats = [
    "targets",
    "receptions",
    "receiving_yards",
    "target_share",
    "receiving_air_yards",
    "air_yards_share",
    "receiving_tds",
    "receiving_yards_after_catch",
]

all_match = True

# Checking Rec Stats
for stat in receiving_stats:
    for suffix in ["baseline", "avg_3", "avg_5"]:
        column = f"{stat}_{suffix}"

        comparison = raw_df[
            ["player_id", "season", "week", column]
        ].merge(
            saved_df[
                ["player_id", "season", "week", column]
            ],
            on=["player_id", "season", "week"],
            how="inner",
            validate="one_to_one",
            suffixes=("_generated", "_saved"),
        )

        matches = np.isclose(
            comparison[f"{column}_generated"],
            comparison[f"{column}_saved"],
            equal_nan=True,
        )

        mismatch_count = (~matches).sum()

        print(
            f"{column}: "
            f"{'PASS' if mismatch_count == 0 else 'FAIL'} "
            f"({mismatch_count} mismatches)"
        )

        if mismatch_count != 0:
            all_match = False

trend_columns = [
    "targets_trend",
    "target_share_trend",
    "rec_yards_trend",
    "rec_air_yards_trend",
]

for column in trend_columns:
    comparison = raw_df[
        ["player_id", "season", "week", column]
    ].merge(
        saved_df[
            ["player_id", "season", "week", column]
        ],
        on=["player_id", "season", "week"],
        how="inner",
        validate="one_to_one",
        suffixes=("_generated", "_saved"),
    )

    matches = np.isclose(
        comparison[f"{column}_generated"],
        comparison[f"{column}_saved"],
        equal_nan=True,
    )

    mismatch_count = (~matches).sum()

    print(
        f"{column}: "
        f"{'PASS' if mismatch_count == 0 else 'FAIL'} "
        f"({mismatch_count} mismatches)"
    )

    if mismatch_count != 0:
        all_match = False
print("\nAll receiving features match:", all_match)

# RB Features Testing
rb_feature_columns = [
    "rb_carries_baseline",
    "rb_carries_avg_3",
    "rb_carries_avg_5",

    "rb_rushing_yards_baseline",
    "rb_rushing_yards_avg_3",
    "rb_rushing_yards_avg_5",

    "rb_rushing_tds_baseline",
    "rb_rushing_tds_avg_3",
    "rb_rushing_tds_avg_5",

    "rb_opportunities_baseline",
    "rb_opportunities_avg_3",
    "rb_opportunities_avg_5",

    "rb_carries_trend",
    "rb_rushing_yards_trend",
    "rb_opportunities_trend",
]

rb_all_match = True

for column in rb_feature_columns:
    comparison = raw_df[
        ["player_id", "season", "week", column]
    ].merge(
        saved_df[
            ["player_id", "season", "week", column]
        ],
        on=["player_id", "season", "week"],
        how="inner",
        validate="one_to_one",
        suffixes=("_generated", "_saved"),
    )

    matches = np.isclose(
        comparison[f"{column}_generated"],
        comparison[f"{column}_saved"],
        equal_nan=True,
    )

    mismatch_count = (~matches).sum()

    print(
        f"{column}: "
        f"{'PASS' if mismatch_count == 0 else 'FAIL'} "
        f"({mismatch_count} mismatches)"
    )

    if mismatch_count != 0:
        rb_all_match = False

print("\nAll RB features match:", rb_all_match)

qb_feature_columns = [
    "completions_baseline",
    "completions_avg_3",
    "completions_avg_5",

    "attempts_baseline",
    "attempts_avg_3",
    "attempts_avg_5",

    "passing_yards_baseline",
    "passing_yards_avg_3",
    "passing_yards_avg_5",

    "passing_tds_baseline",
    "passing_tds_avg_3",
    "passing_tds_avg_5",

    "passing_interceptions_baseline",
    "passing_interceptions_avg_3",
    "passing_interceptions_avg_5",

    "passing_air_yards_baseline",
    "passing_air_yards_avg_3",
    "passing_air_yards_avg_5",

    "passing_first_downs_baseline",
    "passing_first_downs_avg_3",
    "passing_first_downs_avg_5",

    "qb_carries_baseline",
    "qb_carries_avg_3",
    "qb_carries_avg_5",

    "qb_rushing_yards_baseline",
    "qb_rushing_yards_avg_3",
    "qb_rushing_yards_avg_5",

    "qb_rushing_tds_baseline",
    "qb_rushing_tds_avg_3",
    "qb_rushing_tds_avg_5",

    "attempts_trend",
    "passing_yards_trend",
    "passing_air_yards_trend",
    "qb_carries_trend",
    "qb_rushing_yards_trend",
]

qb_all_match = True

for column in qb_feature_columns:
    comparison = raw_df[
        ["player_id", "season", "week", column]
    ].merge(
        saved_df[
            ["player_id", "season", "week", column]
        ],
        on=["player_id", "season", "week"],
        how="inner",
        validate="one_to_one",
        suffixes=("_generated", "_saved"),
    )

    matches = np.isclose(
        comparison[f"{column}_generated"],
        comparison[f"{column}_saved"],
        equal_nan=True,
    )

    mismatch_count = (~matches).sum()

    print(
        f"{column}: "
        f"{'PASS' if mismatch_count == 0 else 'FAIL'} "
        f"({mismatch_count} mismatches)"
    )

    if mismatch_count != 0:
        qb_all_match = False

print("\nAll QB features match:", qb_all_match)

# Kicker Features Testing
kicker_feature_columns = [
    "fg_att_baseline",
    "fg_att_avg_3",
    "fg_att_avg_5",

    "fg_made_baseline",
    "fg_made_avg_3",
    "fg_made_avg_5",

    "fg_long_baseline",
    "fg_long_avg_3",
    "fg_long_avg_5",

    "fg_made_50_59_baseline",
    "fg_made_50_59_avg_3",
    "fg_made_50_59_avg_5",

    "pat_att_baseline",
    "pat_att_avg_3",
    "pat_att_avg_5",

    "pat_made_baseline",
    "pat_made_avg_3",
    "pat_made_avg_5",

    "fg_att_trend",
    "fg_made_trend",
]

kicker_all_match = True

for column in kicker_feature_columns:
    comparison = raw_df[
        ["player_id", "season", "week", column]
    ].merge(
        saved_df[
            ["player_id", "season", "week", column]
        ],
        on=["player_id", "season", "week"],
        how="inner",
        validate="one_to_one",
        suffixes=("_generated", "_saved"),
    )

    matches = np.isclose(
        comparison[f"{column}_generated"],
        comparison[f"{column}_saved"],
        equal_nan=True,
    )

    mismatch_count = (~matches).sum()

    print(
        f"{column}: "
        f"{'PASS' if mismatch_count == 0 else 'FAIL'} "
        f"({mismatch_count} mismatches)"
    )

    if mismatch_count != 0:
        kicker_all_match = False

print("\nAll kicker features match:", kicker_all_match)

# Matchup Features Testing
matchup_feature_columns = [
    "opp_points_allowed_avg_3",
    "opp_points_allowed_avg_5",
    "opp_points_allowed_trend",
]

for column in matchup_feature_columns:

    comparison = raw_df[
        [
            "player_id",
            "season",
            "week",
            "position",
            "opponent_team",
            column,
        ]
    ].merge(
        saved_df[
            [
                "player_id",
                "season",
                "week",
                column,
            ]
        ],
        on=[
            "player_id",
            "season",
            "week",
        ],
        how="inner",
        validate="one_to_one",
        suffixes=("_generated", "_saved"),
    )

    comparison["match"] = np.isclose(
        comparison[f"{column}_generated"],
        comparison[f"{column}_saved"],
        equal_nan=True,
    )

    print(f"\n{column}")

    print("Total rows:", len(comparison))

    print(
        "Generated nulls:",
        comparison[f"{column}_generated"].isna().sum()
    )

    print(
        "Saved nulls:",
        comparison[f"{column}_saved"].isna().sum()
    )

    print("\nMismatches by position:")

    print(
        comparison.loc[
            ~comparison["match"]
        ]
        .groupby("position")
        .size()
        .sort_values(ascending=False)
    )

    print("\nFirst mismatches:")

    print(
        comparison.loc[
            ~comparison["match"],
            [
                "player_id",
                "season",
                "week",
                "position",
                "opponent_team",
                f"{column}_generated",
                f"{column}_saved",
            ]
        ].head(10)
    )