import pandas as pd


# Create a temporary row representing the upcoming game we want to predict.
# This row contains information known before the game, while game stats
# such as targets, yards, touchdowns, etc. remain NaN.
def add_inference_row(
    df,
    player_id,
    position,
    team,
    opponent_team,
    season,
    week,
):
    df = df.copy()

    # Basic information we already know before the upcoming game
    inference_row = {
        "player_id": player_id,
        "position": position,
        "team": team,
        "opponent_team": opponent_team,
        "season": season,
        "week": week,
        "season_type": "REG",
    }

    # Convert the synthetic row into a DataFrame so it can be
    # appended to the player's historical game data
    inference_df = pd.DataFrame([inference_row])

    df = pd.concat(
        [df, inference_df],
        ignore_index=True,
    )

    # Keep player games in chronological order because our rolling
    # feature functions rely on shift() to look at previous games
    df = df.sort_values(
        ["player_id", "season", "week"]
    ).reset_index(drop=True)

    return df


# Add rookie/veteran status information to the synthetic inference row.
# These values are needed by the early-season baseline logic.
def add_inference_player_status(
    df,
    player_id,
    season,
    week,
    rookie_season,
):
    df = df.copy()

    # Look only at seasons before the season we are trying to predict.
    # This prevents the synthetic row itself from being treated as history.
    player_history = df[
        (df["player_id"] == player_id) &
        (df["season"] < season)
    ]

    # Find the most recent season in which the player appeared
    if player_history.empty:
        last_active_season = float("nan")
    else:
        last_active_season = (
            player_history["season"].max()
        )

    # Calculate how many seasons have passed since the player
    # was last active
    if pd.isna(last_active_season):
        season_gap = float("nan")
    else:
        season_gap = season - last_active_season

    # Player is a rookie when the target season is their rookie season
    is_rookie = int(
        season == rookie_season
    )

    # A veteran after a gap is someone who is past their rookie season
    # and either has no usable previous-season history or missed
    # at least one full season before returning
    is_veteran_after_gap = int(
        (season > rookie_season)
        and (
            pd.isna(last_active_season)
            or season_gap > 1
        )
    )

    # Locate the exact synthetic row we created for this prediction
    inference_mask = (
        (df["player_id"] == player_id) &
        (df["season"] == season) &
        (df["week"] == week)
    )

    # Attach the player's status information to the synthetic row
    df.loc[
        inference_mask,
        "rookie_season"
    ] = rookie_season

    df.loc[
        inference_mask,
        "last_active_season"
    ] = last_active_season

    df.loc[
        inference_mask,
        "season_gap"
    ] = season_gap

    df.loc[
        inference_mask,
        "is_rookie"
    ] = is_rookie

    df.loc[
        inference_mask,
        "is_veteran_after_gap"
    ] = is_veteran_after_gap

    return df