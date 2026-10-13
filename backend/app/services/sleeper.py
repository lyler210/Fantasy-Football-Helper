import requests
import nflreadpy as nfl

BASE_URL = "https://api.sleeper.app/v1"

def get_user(username):
    response = requests.get(
        f"{BASE_URL}/user/{username}"
    )

    response.raise_for_status()

    return response.json()

def get_user_leagues(user_id, season):
    response = requests.get(
        f"{BASE_URL}/user/{user_id}/leagues/nfl/{season}"
    )

    response.raise_for_status()

    return response.json()

def get_user_roster(user_id, league_id):
    response = requests.get(
        f"{BASE_URL}/league/{league_id}/rosters"
    )
    response.raise_for_status()
    rosters = response.json()
    for roster in rosters:
        if roster["owner_id"] == user_id:
            return roster
        
    raise ValueError(
        f"No roster found for user {user_id} "
        f"in league {league_id}"
    )

def get_gsis_ids_from_sleeper(sleeper_ids):
    # load cross-platform player ids to pandas table
    player_ids = nfl.load_ff_playerids().to_pandas()

    # changing from float to int to string
    player_ids["sleeper_id_normalized"] = (
        player_ids["sleeper_id"]
        .astype("Int64")
        .astype(str)
    )

    mappings = {}

    for sleeper_id in sleeper_ids:
        # find row with matching sleeper player id
        match = player_ids[
            player_ids["sleeper_id_normalized"] == str(sleeper_id)
        ]

        if match.empty:
            mappings[str(sleeper_id)] = None
        else:
            # adding matched gsis id to list
            mappings[str(sleeper_id)] = (match.iloc[0]["gsis_id"])
    
    # return nflverse/gsis player id
    return mappings

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
    # Loading cross-platform player Id and nfl player metadata
    # player_ids lets us convert sleeper ids -> gsis ids
    # players gives us info for position and team
    player_ids = nfl.load_ff_playerids().to_pandas()
    players = nfl.load_players().to_pandas()

    # sleeper ids stored as floats in nflverse mapping table
    player_ids["sleeper_id_normalized"] = (
        player_ids["sleeper_id"].astype("Int64").astype(str)
    )

    # Store prediction-time information for every roster player
    # each team will contain everything add_inference_row() needs
    inference_inputs = []

    for sleeper_id in sleeper_ids:
        # Skip team defenses for now
        if not str(sleeper_id).isdigit():
            continue
        
        # Find sleeper player in the cross-platform id table
        id_match = player_ids[
            player_ids["sleeper_id_normalized"] == str(sleeper_id)
        ]

        # Skip player if no sleeper -> gsis id mapping exists
        if id_match.empty:
            continue
        
        # Extract gsis id used by nfl verse
        gsis_id = id_match.iloc[0]["gsis_id"]

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

        # Find player's opponent and target season/week
        opponent = find_opponent(team, season, week, schedule_df)

        # Store everything needed to later create the players'
        # synthetic inference row
        inference_inputs.append(
            {
                "player_id": gsis_id,
                "position": position,
                "team": team,
                "opponent_team": opponent,
                "season": season,
                "week": week
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