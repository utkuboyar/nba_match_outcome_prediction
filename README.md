# NBA Game Outcome Prediction
## Model Architecture

![nba_net_drawio](https://user-images.githubusercontent.com/73584680/219902916-393712aa-8aa1-4306-aea2-a747b6e3fc4f.png)

The prediction model is a CNN. The input to the network consists of three parts: player statistics for the most recent games, a binary tensor indicating whether the teams won or lost the most recent games (called "momentum") and bookmaker odds for the home team. The rationale behind the use of convolutional layers is that they serve as aggregators with learnable weights. The tensor for player level statistics are in the form where two channels (that each one is for a team), statistics, top players (in terms of the number of minutes played during the most recent games), and the most recent games are the dimensions. During the first (grouped) convolution, the most recent statistics for each player are aggregated separately for the teams for two times (so four output channels in total). Then the resulting tensors of aggregated player level statistics are further aggregated (again separately for the teams) into team level statistics. A similar procedure is applied for the momentum feature as well. Finally, the resulting team level statistics, aggregated momentum and odds features are combined in a feature vector, which is fed into the fully connected layers.

## Data Collection
The player statistics for the training data were gathered from the official NBA website using nba_api and the odds were scraped from several bookmaker sites. Then they were joined into a custom dataset.

## Portfolio Optimization
The second part of the project aims to optimize the portfolio for a bettor. This is done with a genetic algorithm by optimizing the Sharpe ratio for the given predicted probabilities for the game outcomes and bookmaker odds. To improve the overall performance of the portfolio optimization pipeline, based on the simulations done, some additional rules were added for whether or not to accept the bet with optimized bet ratios.
