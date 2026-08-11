## Using nflreadpy
This python package receives data from nflverse repos, which provides easy access to NFL data with caching, progress tracking, and modern Python conventions. Includes fast data loading with Polars DataFrames.

## Why use PPR points
The app will initially target PPR leagues, so fantasy_points_ppr will be the prediction target.

Future versions should calculate points from each league's actual Sleeper scoring settings

## Using previous-game rolling statistics
Current-week statistics would not be known before the game and would cause data leakage. Using stats from previous weeks to represent the recent form.

## Separating Player and D/ST Prediction Pipelines
The MVP will focus on QB, RB, WR, TE, and K.
D/ST will require a separate team-level prediction pipeline because fantasy defenses represent an entire team's defensive performance rather than an individual defensive player.
I'll postpone D/ST until after the MVP works so that I can first learn and validate the core ML workflow without adding this complexity to it.

## What is data leakage?
Data leakage is when informating unavailable at prediction time accidentally being included in the model's inputs. For example: Trying to predict a player's week 5 fantasy points, but using only their stats from week 5.

