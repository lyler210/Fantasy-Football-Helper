import nflreadpy as nfl
from app.services.sleeper import (
    get_user,
    get_user_leagues,
    get_user_roster,
    get_gsis_ids_from_sleeper,
    find_opponent,
    build_roster_inference_inputs,
    split_bye_week_players
)

username = "lyler210"

user = get_user(username)
print("\nUSER:")
print(user)

leagues = get_user_leagues(
    user_id=user["user_id"],
    season=2026
)

print("\nLEAGUES:")

for league in leagues:
    print(league["name"], "|", league["league_id"])

chosen_league_id = "1311990336242847744"
chosen_roster = get_user_roster(user["user_id"], chosen_league_id)
print(f"Roster: {chosen_roster["players"]}")
print(f"Starters: {chosen_roster["starters"]}")

mappings = get_gsis_ids_from_sleeper(
    chosen_roster["players"]
)

print(mappings)
schedule_df = nfl.load_schedules(2026).to_pandas()
print(find_opponent("SF", 2026, 6, schedule_df))

inference_inputs = build_roster_inference_inputs(
    sleeper_ids=chosen_roster["players"],
    season=2026,
    week=6,
    schedule_df=schedule_df
)

print("\nINFERENCE INPUTS:")

for player in inference_inputs:
    print(player)

players_to_predict, bye_week_players = split_bye_week_players(
    inference_inputs
)

print("\nPLAYERS TO PREDICT:")
for player in players_to_predict:
    print(player)

print("\nBYE WEEK PLAYERS:")
for player in bye_week_players:
    print(player)