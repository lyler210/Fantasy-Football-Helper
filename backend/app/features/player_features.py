import pandas as pd


# Early Season Baseline Helper
def create_early_season_baseline(df, stat, positions):

    # Only use columns needed for calculating baseline
    baseline_df = df[
        [
            'player_id',
            'season',
            'position',
            'last_active_season',
            'is_rookie',
            'is_veteran_after_gap'
        ]
    ].copy()

    # Keep original row order
    baseline_df['row_id'] = df.index

    # Veterans

    # Average stat for each player's season
    player_season_avg = (
        df[df['position'].isin(positions)]
        .groupby(
            ['player_id', 'season'],
            as_index=False
        )[stat]
        .mean()
        .rename(
            columns={
                'season': 'last_active_season',
                stat: 'veteran_last_active_season_avg'
            }
        )
    )

    baseline_df = baseline_df.merge(
        player_season_avg,
        on=['player_id', 'last_active_season'],
        how='left',
        validate='many_to_one'
    )

    # Rookies

    rookie_historical_avg_rows = []

    for season in range(
        int(df['season'].min()) + 1,
        int(df['season'].max()) + 1
    ):

        # Only use rookies from seasons before the current season
        historical_rookies = df[
            (df['season'] < season) &
            (df['is_rookie'] == 1) &
            (df['position'].isin(positions))
        ]

        position_historical_avg = (
            historical_rookies
            .groupby('position')[stat]
            .mean()
        )

        for position, historical_avg in position_historical_avg.items():
            rookie_historical_avg_rows.append({
                'season': season,
                'position': position,
                'rookie_historical_avg': historical_avg
            })
    
    rookie_historical_avg = pd.DataFrame(
        rookie_historical_avg_rows
    )

    baseline_df = baseline_df.merge(
        rookie_historical_avg,
        on=['season', 'position'],
        how='left',
        validate='many_to_one'
    )

    # Historical Position Average 
    position_historical_avg_rows = []

    for season in range(
        int(df['season'].min()) + 1,
        int(df['season'].max()) + 1
    ):
        
        # Only use seasons before current season
        historical_players = df[(df['season'] < season) & (df['position'].isin(positions))]

        position_historical_avg = (historical_players.groupby('position')[stat].mean())

        for position, historical_avg in position_historical_avg.items():
            position_historical_avg_rows.append({
                'season': season,
                'position': position,
                'position_historical_avg': historical_avg
            })
    
    position_historical_avg_df = pd.DataFrame(position_historical_avg_rows)

    baseline_df = baseline_df.merge(
        position_historical_avg_df,
        on=['season', 'position'],
        how='left',
        validate='many_to_one'
    )

    # Choosing Baseline

    baseline_df['baseline'] = float('nan')

    # Rookies use historical average for rookies at their position
    baseline_df.loc[
        baseline_df['is_rookie'] == 1,
        'baseline'
    ] = baseline_df.loc[
        baseline_df['is_rookie'] == 1,
        'rookie_historical_avg'
    ]

    # Veterans after a season gap use their last active-season average
    baseline_df.loc[
        baseline_df['is_veteran_after_gap'] == 1,
        'baseline'
    ] = baseline_df.loc[
        baseline_df['is_veteran_after_gap'] == 1,
        'veteran_last_active_season_avg'
    ]

    # If veteran's last active season is outside dataset,
    # Use historical average for their position

    missing_veteran_baseline = (
        (baseline_df['is_veteran_after_gap'] == 1) &
        (baseline_df['baseline'].isna())
    )

    baseline_df.loc[
        missing_veteran_baseline,
        'baseline'
    ] = baseline_df.loc[
        missing_veteran_baseline,
        'position_historical_avg'
    ]

    # Restore original DF order
    baseline_df = baseline_df.sort_values('row_id')

    return baseline_df['baseline'].set_axis(df.index)


# Rolling Average Helper
# Calculate average stat from up to the previous N eligible games
def rolling_avg_with_last_season(df, stat, window, baseline=None):

    past_stats = []

    for games_back in range(1, window + 1):

        # Get player's stat from N games ago
        previous_value = (
            df.groupby('player_id')[stat]
            .shift(games_back)
        )

        # Get season that previous game occurred in
        previous_season = (
            df.groupby('player_id')['season']
            .shift(games_back)
        )

        # Only allow games from current season or immediately previous season
        valid_history = (
            (previous_season == df['season']) |
            (previous_season == df['season'] - 1)
        )

        # Keep the stat only if it comes from valid recent history
        eligible_value = previous_value.where(
            valid_history
        )

        # If no valid recent game exists, use the player's baseline
        if baseline is not None:
            eligible_value = eligible_value.fillna(
                baseline
            )

        past_stats.append(
            eligible_value
        )

    # Put previous game stats side-by-side and average across them
    return pd.concat(
        past_stats,
        axis=1
    ).mean(axis=1)

# Rec Features Helper
def add_receiving_features(df):
    receiving_positions = ["RB", "WR", "TE"]

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

    receiving_features = {}

    for stat in receiving_stats:

        baseline = create_early_season_baseline(
            df,
            stat,
            receiving_positions
        )

        receiving_features[f"{stat}_baseline"] = baseline

        receiving_features[f"{stat}_avg_3"] = (
            rolling_avg_with_last_season(
                df,
                stat,
                3,
                baseline
            )
        )

        receiving_features[f"{stat}_avg_5"] = (
            rolling_avg_with_last_season(
                df,
                stat,
                5,
                baseline
            )
        )

    feature_df = pd.DataFrame(
        receiving_features,
        index=df.index
    )

    df = pd.concat(
        [df, feature_df],
        axis=1
    )

    df["targets_trend"] = (
        df["targets_avg_3"] - df["targets_avg_5"]
    )

    df["target_share_trend"] = (
        df["target_share_avg_3"] - df["target_share_avg_5"]
    )

    df["rec_yards_trend"] = (
        df["receiving_yards_avg_3"] - df["receiving_yards_avg_5"]
    )

    df["rec_air_yards_trend"] = (
        df["receiving_air_yards_avg_3"] - df["receiving_air_yards_avg_5"]
    )

    return df

# RB Features Helper
def add_rb_features(df):
    df = df.copy()

    df['opportunities'] = df['carries'] + df['targets']

    rb_stats = {
        "carries": "rb_carries",
        "rushing_yards": "rb_rushing_yards",
        "rushing_tds": "rb_rushing_tds",
        "opportunities": "rb_opportunities",
    }

    rb_features = {}

    for stat, feature_name in rb_stats.items():

        baseline = create_early_season_baseline(
            df,
            stat,
            ["RB"]
        )

        rb_features[f"{feature_name}_baseline"] = baseline

        rb_features[f"{feature_name}_avg_3"] = (
            rolling_avg_with_last_season(
                df,
                stat,
                3,
                baseline
            )
        )

        rb_features[f"{feature_name}_avg_5"] = (
            rolling_avg_with_last_season(
                df,
                stat,
                5,
                baseline
            )
        )

    feature_df = pd.DataFrame(
        rb_features,
        index=df.index
    )

    df = pd.concat(
        [df, feature_df],
        axis=1
    )

    df["rb_carries_trend"] = (
        df["rb_carries_avg_3"] - df["rb_carries_avg_5"]
    )

    df["rb_rushing_yards_trend"] = (
        df["rb_rushing_yards_avg_3"] - df["rb_rushing_yards_avg_5"]
    )

    df["rb_opportunities_trend"] = (
        df["rb_opportunities_avg_3"] - df["rb_opportunities_avg_5"]
    )

    return df

# QB Features Helper
def add_qb_features(df):
    df = df.copy()

    qb_stats = {
        "completions": "completions",
        "attempts": "attempts",
        "passing_yards": "passing_yards",
        "passing_tds": "passing_tds",
        "passing_interceptions": "passing_interceptions",
        "passing_air_yards": "passing_air_yards",
        "passing_first_downs": "passing_first_downs",

        # Prefix rushing stats so they don't collide with RB features
        "carries": "qb_carries",
        "rushing_yards": "qb_rushing_yards",
        "rushing_tds": "qb_rushing_tds",
    }

    qb_features = {}

    for stat, feature_name in qb_stats.items():

        baseline = create_early_season_baseline(
            df,
            stat,
            ["QB"]
        )

        qb_features[f"{feature_name}_baseline"] = baseline

        qb_features[f"{feature_name}_avg_3"] = (
            rolling_avg_with_last_season(
                df,
                stat,
                3,
                baseline
            )
        )

        qb_features[f"{feature_name}_avg_5"] = (
            rolling_avg_with_last_season(
                df,
                stat,
                5,
                baseline
            )
        )

    feature_df = pd.DataFrame(
        qb_features,
        index=df.index
    )

    df = pd.concat(
        [df, feature_df],
        axis=1
    )

    df["attempts_trend"] = (
        df["attempts_avg_3"]
        - df["attempts_avg_5"]
    )

    df["passing_yards_trend"] = (
        df["passing_yards_avg_3"]
        - df["passing_yards_avg_5"]
    )

    df["passing_air_yards_trend"] = (
        df["passing_air_yards_avg_3"]
        - df["passing_air_yards_avg_5"]
    )

    df["qb_carries_trend"] = (
        df["qb_carries_avg_3"]
        - df["qb_carries_avg_5"]
    )

    df["qb_rushing_yards_trend"] = (
        df["qb_rushing_yards_avg_3"]
        - df["qb_rushing_yards_avg_5"]
    )

    return df

# Kicker Features Helper
def add_kicker_features(df):
    df = df.copy()

    kicker_stats = [
        "fg_att",
        "fg_made",
        "fg_long",
        "fg_made_50_59",
        "pat_att",
        "pat_made",
    ]

    kicker_features = {}

    for stat in kicker_stats:

        baseline = create_early_season_baseline(
            df,
            stat,
            ["K"]
        )

        kicker_features[f"{stat}_baseline"] = baseline

        kicker_features[f"{stat}_avg_3"] = (
            rolling_avg_with_last_season(
                df,
                stat,
                3,
                baseline
            )
        )

        kicker_features[f"{stat}_avg_5"] = (
            rolling_avg_with_last_season(
                df,
                stat,
                5,
                baseline
            )
        )

    feature_df = pd.DataFrame(
        kicker_features,
        index=df.index
    )

    df = pd.concat(
        [df, feature_df],
        axis=1
    )

    df["fg_att_trend"] = (
        df["fg_att_avg_3"]
        - df["fg_att_avg_5"]
    )

    df["fg_made_trend"] = (
        df["fg_made_avg_3"]
        - df["fg_made_avg_5"]
    )

    return df