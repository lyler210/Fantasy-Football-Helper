## Row layout for training
Each row will represent a player entering a specific game/week. Predictor features must contain only information that would have been available BEFORE the game, such as historical player performance, recent usage, season-to-date statistics, opponent information, and matchup context. The player's actual fantasy points from that game will serve as the training target.
Example: A row for a WR going into a Week 8 game would contain his average receiving yards from weeks 5-7, average target share from Weeks 5-7, and his week 8 opponent
ONE EXCEPTION: Including Week 8 fantasy points. For a training row, theres Features (X) = information known before Week 8, and Target(Y) = what actually happened in Week 8 which is 'fantasy_points_ppr'.
This is crucial because it is what the model actually learns from (its like an answer key).

## First ML approach - Supervised learning
I will first explore using supervised learning with a regression model as the initial ML approach. Past games will provide labeled training examples as features (X), and the player's actual fantasy points are the target (Y). The model will predict the expected fantasy points rather than directly classifying what players to Start vs Sit. Those recommendations will be determined by comparing player predictions within the user's roster.
Current hierarchy: ML -> Supervised Learning -> Regression -> Predict fantasy points

## Feature Engineering - Rolling Averages
1. Why not use the target week's stats?
A: This will lead to data leakage because those stats are not known before the game, causing the model to fail when it's actually used
2. Why choose last 3 game averages?
A: This will show the current performance of the player
3. Why chose last 5 game averages?
A: This will reveal how the player has generally been performing over a longer-term baseline. 
4. What will comparing these two numbers do?
A: Comparing the 3 game average to this will also allow the model to detect if there are any changes in recent performance.

## Season boundaries for rolling averages
For the initial model, the rolling averages will be reset at the start of each NFL season. The early-season predictions will only be based on whatever information is already available rather than trying to pull from previous season games. This avoids carrying over a potentially outdated context from an old coordinator, roster, etc. This also helps prevent numbers weeks 17/18 from affecting the model, where teams that are locked in for the playoffs are more likely to sit their starters to prevent injury.
To sort by player id were using group by which prevents players from mixing their data, and shift(1) to exclude the current game

# Historical Data Usage
I will start off by using player statistics from the 2019-2025 seasons to provide the model with a larger training dataset. Older data may become less representative of the current NFL, so the historical range can later be treated as a modeling decision and evaluated through validation performance.

# Selecting Trend Features
Initially using these trend features for WRs - targets, target share, receiving yards, and receiving air yards. A Trend is defined as 3-game rolling average - 5-game rolling average, where positive values will indicate recent performance/usage above the broader 5 game baseline and negative values indicate a decline

# Splitting up WR and TE
At this point I have finished finding all of the models that I want to use for each position before doing the final evaluation. But I want to separate the WR and TE dataset because their offensive roles differ depending on the team. An elite TE could see the same amount of opportunities as a WR2 or even a WR1, while other TEs have less opportunity.

# Models for positions
WR - Linear Regression
TE - Linear Regression
RB - Linear Regression
QB - Tuned/Grid-searched Random Forest
K - Linear Regression