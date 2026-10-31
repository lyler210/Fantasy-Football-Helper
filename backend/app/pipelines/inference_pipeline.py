import nflreadpy as nfl
import pandas as pd

from app.features.player_features import (
    add_receiving_features,
    add_rb_features,
    add_qb_features,
    add_kicker_features,
)
from app.features.matchup_features import add_matchup_features

from app.models.predictor import predict_player, FEATURE_CONFIG

from app.features.inference_features import add_inference_row, add_inference_player_status

from app.services.sleeper import get_gsis_ids_from_sleeper

# opponent helper function
def find_opponent(team, season, week, schedule_df):
    # Find the game for this team in the target season/week
    game = schedule_df[
        (schedule_df["season"] == season)
        & (schedule_df["week"] == week)
        & (
            (schedule_df["home_team"] == team)
            | (schedule_df["away_team"] == team)
        )
    ]

    # No game found means the team is on a bye
    if game.empty:
        return None

    game = game.iloc[0]

    # Return whichever team is on the opposite side
    if game["home_team"] == team:
        return game["away_team"]
    
    return game["home_team"]

def build_roster_inference_inputs(sleeper_ids, season, week, schedule_df):
    # Convert Sleeper IDs to GSIS IDs
    gsis_mappings = get_gsis_ids_from_sleeper(sleeper_ids)

    # Load NFL player Metadata for position and team look up
    players = nfl.load_players().to_pandas()

    # Store prediction-time information for every roster player
    inference_inputs = []

    for sleeper_id in sleeper_ids:
        # Skip team defenses for now
        if not str(sleeper_id).isdigit():
            continue
        
        # Get mapped GSIS ID for Sleeper player
        gsis_id = gsis_mappings.get(str(sleeper_id))
        
        # Skip players that could not be mapped
        if gsis_id is None:
            continue
        
        # find player's metadata using gsis id
        player_match = players[
            players["gsis_id"] == gsis_id
        ]

        # Skip player if no metadata record can be found
        if player_match.empty:
            continue

        player = player_match.iloc[0]
        player_name = player["display_name"]

        position = player["position"]
        team = player["latest_team"]
        rookie_season = player["rookie_season"]

        # Find player's opponent and target season/week
        opponent = find_opponent(team, season, week, schedule_df)

        # Store everything needed to create synthetic inference row
        inference_inputs.append(
            {
                "player_id": gsis_id,
                "player_name": player_name,
                "position": position,
                "team": team,
                "opponent_team": opponent,
                "season": season,
                "week": week,
                "rookie_season": rookie_season
            }
        )

    return inference_inputs
    
# Bye Week Helper
def split_bye_week_players(inference_inputs):
    players_to_predict = []
    bye_week_players = []

    for player in inference_inputs:
        if player["opponent_team"] is None:
            bye_week_players.append(player)
        else:
            players_to_predict.append(player)
        
    return players_to_predict, bye_week_players

def add_roster_inference_rows(df, players_to_predict):
    # Copy OG dataframe
    updated_df = df.copy()

    # Add one synthetic upcoming-game row for each playable roster player
    for player in players_to_predict:
        updated_df = add_inference_row(
            updated_df,
            player_id=player["player_id"],
            position=player["position"],
            team=player["team"],
            opponent_team=player["opponent_team"],
            season=player["season"],
            week=player["week"],
        )

        updated_df = add_inference_player_status(
            updated_df,
            player_id=player["player_id"],
            season=player["season"],
            week=player["week"],
            rookie_season=player["rookie_season"],
        )
    
    return updated_df

def prepare_historical_data(df):
    # Load player metadata
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

    # Add rookie season to historical game rows
    df = df.merge(
        player_metadata,
        on="player_id",
        how="left",
    )

    # Create one row per player-season
    player_seasons = (
        df[
            [
                "player_id",
                "season",
            ]
        ]
        .dropna(subset=["player_id"])
        .drop_duplicates()
        .sort_values(
            [
                "player_id",
                "season",
            ]
        )
    )

    # Find the previous season each player was active
    player_seasons["last_active_season"] = (
        player_seasons
        .groupby("player_id")["season"]
        .shift(1)
    )

    # Calculate gap between active seasons
    player_seasons["season_gap"] = (
        player_seasons["season"]
        - player_seasons["last_active_season"]
    )

    # Attach season-status information to game rows
    df = df.merge(
        player_seasons,
        on=[
            "player_id",
            "season",
        ],
        how="left",
    )

    # Mark rookies
    df["is_rookie"] = (
        df["season"]
        == df["rookie_season"]
    ).astype(int)

    # Mark veterans returning after a gap
    df["is_veteran_after_gap"] = (
        (df["season"] > df["rookie_season"])
        &
        (
            df["last_active_season"].isna()
            | (df["season_gap"] > 1)
        )
    ).astype(int)

    # Keep history chronological for shift-based features
    df = df.sort_values(
        [
            "player_id",
            "season",
            "week",
        ]
    ).reset_index(drop=True)

    return df


# Build Roster Predictions
def build_roster_predictions(df, players_to_predict):
    # Add one fully prepared synthetic row for each playable roster player
    updated_df = add_roster_inference_rows(
        df,
        players_to_predict
    )

    # Run the same production feature engineering used during training
    updated_df = add_receiving_features(updated_df)
    updated_df = add_rb_features(updated_df)
    updated_df = add_qb_features(updated_df)
    updated_df = add_kicker_features(updated_df)
    updated_df = add_matchup_features(updated_df)

    prediction_rows = []

    for player in players_to_predict:
        # Find this player's synthetic target-week row
        player_row = updated_df[
            (updated_df["player_id"] == player["player_id"])
            & (updated_df["season"] == player["season"])
            & (updated_df["week"] == player["week"])
        ].copy()

        if player_row.empty:
            continue

        player_row = player_row.iloc[0]

        position = player["position"]

        # Get the exact feature list expected by this position's model
        required_features = FEATURE_CONFIG[position]["features"]

        # Build the model input from the engineered synthetic row
        player_features = {
            feature: player_row[feature]
            for feature in required_features
        }

        prediction = predict_player(
            position,
            player_features
        )

        # Convert the full synthetic row into a dictionary
        result = player_row.to_dict()
        result["player_name"] = player["player_name"]

        # Add the model prediction to the synthetic row
        result["projected_fantasy_points"] = prediction

        prediction_rows.append(result)

    return pd.DataFrame(prediction_rows)