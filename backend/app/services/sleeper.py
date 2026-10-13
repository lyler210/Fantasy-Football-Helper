import requests
import nflreadpy as nfl

from app.features.inference_features import add_inference_row

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
