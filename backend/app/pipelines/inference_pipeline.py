import nflreadpy as nfl

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

        position = player["position"]
        team = player["latest_team"]
        rookie_season = player["rookie_season"]

        # Find player's opponent and target season/week
        opponent = find_opponent(team, season, week, schedule_df)

        # Store everything needed to create synthetic inference row
        inference_inputs.append(
            {
                "player_id": gsis_id,
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
