# NBA Game Outcome Prediction
## Model Architecture

![nba_net_drawio](https://user-images.githubusercontent.com/73584680/219902916-393712aa-8aa1-4306-aea2-a747b6e3fc4f.png)

The underlying architecture of the prediction model is a CNN. The CNN takes three inputs: player statistics for the most recent games, a binary tensor indicating the momentum of each team (whether they won or lost their recent games), and bookmaker odds for the home team.

The player statistics are organized into a tensor with two channels (one for each team), statistics, top players based on the number of minutes played, and recent games. The convolutional layers in the model serve as aggregators with learnable weights, which are used to aggregate the statistics separately for each team and create team-level statistics.

A similar procedure is applied for the momentum feature, which is also aggregated separately for each team. Finally, the team-level statistics, aggregated momentum, and odds features are combined into a feature vector that is fed into the fully connected layers of the CNN. The fully connected layers then output the predicted outcome of the game.

## Data Collection
The player statistics for the training data were gathered from the official NBA website using nba_api and the odds were scraped from several bookmaker sites. Then they were joined into a custom dataset.

## Portfolio Optimization
The second part of the project aims to optimize a bettor's portfolio using a genetic algorithm that optimizes the Sharpe ratio based on predicted probabilities for game outcomes and bookmaker odds. To improve the overall performance of the portfolio optimization pipeline, additional rules were added to determine whether or not to accept bets based on the optimized bet ratios.
