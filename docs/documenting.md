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

## Row layout for training
Each row will represent a player entering a specific game/week. Predictor features must contain only information that would have been available BEFORE the game, such as historical player performance, recent usage, season-to-date statistics, opponent information, and matchup context. The player's actual fantasy points from that game will serve as the training target.
Example: A row for a WR going into a Week 8 game would contain his average receiving yards from weeks 5-7, average target share from Weeks 5-7, and his week 8 opponent
ONE EXCEPTION: Including Week 8 fantasy points. For a training row, theres Features (X) = information known before Week 8, and Target(Y) = what actually happened in Week 8 which is 'fantasy_points_ppr'.
This is crucial because it is what the model actually learns from (its like an answer key).

## First ML approach - Supervised learning
I will first explore using supervised learning with a regression model as the initial ML approach. Past games will provide labeled training examples as features (X), and the player's actual fantasy points are the target (Y). The model will predict the expected fantasy points rather than directly classifying what players to Start vs Sit. Those recommendations will be determined by comparing player predictions within the user's roster.
Current hierarchy: ML -> Supervised Learning -> Regression -> Predict fantasy points