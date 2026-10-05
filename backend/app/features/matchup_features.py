import pandas as pd

# Calculate points allowed using current and immediately previous season
def rolling_matchup_avg(
    df,
    stat,
    window
):

    past_stats = []

    matchup_group = [
        'opponent_team',
        'position'
    ]

    for games_back in range(1, window + 1):

        previous_value = (
            df.groupby(matchup_group)[stat]
            .shift(games_back)
        )

        previous_season = (
            df.groupby(matchup_group)['season']
            .shift(games_back)
        )

        # Only allow current season or immediately previous season
        valid_history = (
            (previous_season == df['season']) |
            (previous_season == df['season'] - 1)
        )

        past_stats.append(
            previous_value.where(valid_history)
        )

    return pd.concat(
        past_stats,
        axis=1
    ).mean(axis=1)

# Matchup Features Helper
def add_matchup_features(df):
    df = df.copy()

    fantasy_positions = [
        "QB",
        "RB",
        "WR",
        "TE",
        "K",
    ]

    matchup_df = df[
        df["position"].isin(fantasy_positions)
    ].copy()

    # Default scoring for QB/RB/WR/TE
    matchup_df["matchup_points"] = matchup_df["fantasy_points_ppr"]

    # K uses the simplified kicker scoring
    kicker_mask = matchup_df["position"] == "K"

    matchup_df.loc[kicker_mask,"matchup_points"] = (
        3 * matchup_df.loc[kicker_mask, "fg_made"]
        + matchup_df.loc[kicker_mask, "pat_made"]
    )

    # Total points allowedby each opponent to each position that week
    matchup_weekly = (
        matchup_df.groupby(
            [
                "season",
                "week",
                "opponent_team",
                "position"
            ],
            as_index=False,
        )["matchup_points"].sum().rename(
            columns={
            "matchup_points": "points_allowed"
            }
        )
    )

    # Shift() requires chronological order
    matchup_weekly = matchup_weekly.sort_values(
        [
            "opponent_team",
            "position",
            "season",
            "week",
        ]
    ).reset_index(drop=True)

    matchup_weekly["opp_points_allowed_avg_3"] = (
        rolling_matchup_avg(
            matchup_weekly,
            "points_allowed",
            3
        )
    )

    matchup_weekly["opp_points_allowed_avg_5"] = (
        rolling_matchup_avg(
            matchup_weekly,
            "points_allowed",
            5
        )
    )

    matchup_weekly["opp_points_allowed_trend"] = (
        matchup_weekly["opp_points_allowed_avg_3"]
        - matchup_weekly["opp_points_allowed_avg_5"]
    )

    matchup_features = matchup_weekly[
        [
            "season",
            "week",
            "opponent_team",
            "position",
            "opp_points_allowed_avg_3",
            "opp_points_allowed_avg_5",
            "opp_points_allowed_trend",
        ]
    ]

    df = df.merge(
        matchup_features,
        on=[
            "season",
            "week",
            "opponent_team",
            "position",
        ],
        how="left",
        validate="many_to_one",
    )

    return df 

