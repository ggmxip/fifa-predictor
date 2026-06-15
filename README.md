# FIFA World Cup Group Stage Predictor

Hybrid ML + Statistical ensemble model for World Cup match outcome prediction.

## Model Performance (2018 + 2022 Test Set)

| Model | Accuracy | Log Loss | Brier Score |
|-------|----------|----------|-------------|
| Poisson | 62.50% | 0.9225 | 0.1747 |
| Elo | 58.33% | 0.9351 | 0.1819 |
| Hybridensemble | 60.42% | 0.9329 | 0.1823 |
| Random Forest | 51.04% | 1.0408 | 0.2078 |
| Lightgbm | 44.79% | 1.2871 | 0.2421 |
| Xgboost | 36.46% | 1.3266 | 0.2558 |
| Neural Network | 32.29% | 2.4732 | 0.3452 |

## All Group Stage Fixtures (48 Matches)

Predicted probabilities for every 2026 World Cup group stage match.

### Group A

| Match | H Win | Draw | A Win | Prediction |
|-------|-------|------|-------|------------|
| Mexico vs South Korea | 43.0% | 20.5% | 36.4% | **H** |
| Mexico vs Czech Republic | 46.2% | 21.4% | 32.4% | **H** |
| Mexico vs South Africa | 47.7% | 18.9% | 33.5% | **H** |
| South Korea vs Czech Republic | 42.1% | 23.1% | 34.8% | **H** |
| South Korea vs South Africa | 43.4% | 20.7% | 35.9% | **H** |
| Czech Republic vs South Africa | 38.8% | 21.5% | 39.7% | **A** |
### Group B

| Match | H Win | Draw | A Win | Prediction |
|-------|-------|------|-------|------------|
| Switzerland vs Canada | 77.9% | 9.9% | 12.2% | **H** |
| Switzerland vs Qatar | 79.0% | 10.6% | 10.3% | **H** |
| Switzerland vs Bosnia and Herzegovina | 51.2% | 19.1% | 29.7% | **H** |
| Canada vs Qatar | 48.6% | 22.5% | 28.8% | **H** |
| Canada vs Bosnia and Herzegovina | 17.6% | 13.7% | 68.7% | **A** |
| Qatar vs Bosnia and Herzegovina | 15.1% | 13.9% | 71.0% | **A** |
### Group C

| Match | H Win | Draw | A Win | Prediction |
|-------|-------|------|-------|------------|
| Scotland vs Morocco | 25.7% | 25.5% | 48.8% | **A** |
| Scotland vs Brazil | 12.0% | 12.5% | 75.5% | **A** |
| Scotland vs Haiti | 37.0% | 26.0% | 37.0% | **H** |
| Morocco vs Brazil | 19.5% | 21.1% | 59.4% | **A** |
| Morocco vs Haiti | 48.8% | 25.5% | 25.7% | **H** |
| Brazil vs Haiti | 75.5% | 12.5% | 12.0% | **H** |
### Group D

| Match | H Win | Draw | A Win | Prediction |
|-------|-------|------|-------|------------|
| United States vs Australia | 60.5% | 16.1% | 23.3% | **H** |
| United States vs Turkey | 43.9% | 22.3% | 33.8% | **H** |
| United States vs Paraguay | 39.7% | 23.2% | 37.1% | **H** |
| Australia vs Turkey | 27.9% | 18.2% | 53.9% | **A** |
| Australia vs Paraguay | 24.9% | 17.3% | 57.8% | **A** |
| Turkey vs Paraguay | 34.9% | 23.2% | 42.0% | **A** |
### Group E

| Match | H Win | Draw | A Win | Prediction |
|-------|-------|------|-------|------------|
| Germany vs Ecuador | 59.6% | 18.3% | 22.1% | **H** |
| Germany vs Ivory Coast | 64.3% | 15.9% | 19.9% | **H** |
| Germany vs Curacao | 70.2% | 14.2% | 15.6% | **H** |
| Ecuador vs Ivory Coast | 41.4% | 22.5% | 36.2% | **H** |
| Ecuador vs Curacao | 50.2% | 22.7% | 27.1% | **H** |
| Ivory Coast vs Curacao | 49.0% | 22.2% | 28.8% | **H** |
### Group F

| Match | H Win | Draw | A Win | Prediction |
|-------|-------|------|-------|------------|
| Japan vs Netherlands | 12.2% | 13.6% | 74.2% | **A** |
| Japan vs Sweden | 22.2% | 18.9% | 58.8% | **A** |
| Japan vs Tunisia | 63.9% | 17.2% | 18.9% | **H** |
| Netherlands vs Sweden | 55.2% | 21.5% | 23.3% | **H** |
| Netherlands vs Tunisia | 88.8% | 5.7% | 5.5% | **H** |
| Sweden vs Tunisia | 79.6% | 9.8% | 10.6% | **H** |
### Group G

| Match | H Win | Draw | A Win | Prediction |
|-------|-------|------|-------|------------|
| Belgium vs Egypt | 75.6% | 11.5% | 12.9% | **H** |
| Belgium vs Iran | 57.9% | 19.2% | 22.9% | **H** |
| Belgium vs New Zealand | 37.2% | 29.7% | 33.2% | **H** |
| Egypt vs Iran | 22.3% | 20.0% | 57.7% | **A** |
| Egypt vs New Zealand | 16.3% | 20.0% | 63.7% | **A** |
| Iran vs New Zealand | 24.0% | 27.9% | 48.0% | **A** |
### Group H

| Match | H Win | Draw | A Win | Prediction |
|-------|-------|------|-------|------------|
| Spain vs Cape Verde | 65.2% | 15.6% | 19.2% | **H** |
| Spain vs Saudi Arabia | 88.6% | 4.5% | 6.9% | **H** |
| Spain vs Uruguay | 34.2% | 22.3% | 43.4% | **A** |
| Cape Verde vs Saudi Arabia | 70.2% | 14.6% | 15.2% | **H** |
| Cape Verde vs Uruguay | 18.1% | 19.8% | 62.1% | **A** |
| Saudi Arabia vs Uruguay | 6.2% | 6.5% | 87.2% | **A** |
### Group I

| Match | H Win | Draw | A Win | Prediction |
|-------|-------|------|-------|------------|
| France vs Senegal | 37.1% | 21.9% | 41.0% | **A** |
| France vs Iraq | 53.8% | 22.2% | 23.9% | **H** |
| France vs Norway | 53.8% | 22.2% | 23.9% | **H** |
| Senegal vs Iraq | 61.6% | 17.0% | 21.5% | **H** |
| Senegal vs Norway | 61.6% | 17.0% | 21.5% | **H** |
| Iraq vs Norway | 37.0% | 26.0% | 37.0% | **H** |
### Group J

| Match | H Win | Draw | A Win | Prediction |
|-------|-------|------|-------|------------|
| Argentina vs Algeria | 66.0% | 15.7% | 18.3% | **H** |
| Argentina vs Austria | 70.4% | 13.9% | 15.6% | **H** |
| Argentina vs Jordan | 70.4% | 13.9% | 15.6% | **H** |
| Algeria vs Austria | 42.4% | 25.0% | 32.6% | **H** |
| Algeria vs Jordan | 42.4% | 25.0% | 32.6% | **H** |
| Austria vs Jordan | 37.0% | 26.0% | 37.0% | **H** |
### Group K

| Match | H Win | Draw | A Win | Prediction |
|-------|-------|------|-------|------------|
| Portugal vs Congo DR | 64.8% | 16.1% | 19.1% | **H** |
| Portugal vs Uzbekistan | 64.8% | 16.1% | 19.1% | **H** |
| Portugal vs Colombia | 23.5% | 18.2% | 58.3% | **A** |
| Congo DR vs Uzbekistan | 37.0% | 26.0% | 37.0% | **H** |
| Congo DR vs Colombia | 13.0% | 11.1% | 75.9% | **A** |
| Uzbekistan vs Colombia | 13.0% | 11.1% | 75.9% | **A** |
### Group L

| Match | H Win | Draw | A Win | Prediction |
|-------|-------|------|-------|------------|
| England vs Croatia | 38.6% | 24.6% | 36.8% | **H** |
| England vs Ghana | 55.6% | 19.0% | 25.4% | **H** |
| England vs Panama | 86.5% | 5.1% | 8.3% | **H** |
| Croatia vs Ghana | 55.2% | 18.5% | 26.3% | **H** |
| Croatia vs Panama | 87.0% | 4.9% | 8.1% | **H** |
| Ghana vs Panama | 77.9% | 8.9% | 13.2% | **H** |

## Group Stage Simulation

Monte Carlo simulation (500 runs) showing qualification probabilities for each team.

| Group | Team | Qualify % | Win Group % | Avg Points |
|-------|------|-----------|-------------|------------|
| A | Mexico | 63.6% | 40.2% | 4.7 |
|  | South Korea | 54.8% | 26.2% | 4.33 |
|  | Czech Republic | 47.2% | 20.8% | 3.99 |
|  | South Africa | 34.4% | 12.8% | 3.76 |
| B | Switzerland | 90.8% | 67.0% | 6.53 |
|  | Bosnia and Herzegovina | 66.6% | 24.8% | 5.6 |
|  | Canada | 27.4% | 5.2% | 2.87 |
|  | Qatar | 15.2% | 3.0% | 2.16 |
| C | Brazil | 83.2% | 57.4% | 6.51 |
|  | Morocco | 54.6% | 24.2% | 4.3 |
|  | Scotland | 38.6% | 10.4% | 3.01 |
|  | Haiti | 23.6% | 8.0% | 2.97 |
| D | United States | 71.2% | 43.4% | 4.94 |
|  | Turkey | 51.0% | 25.2% | 4.41 |
|  | Paraguay | 49.0% | 21.4% | 4.6 |
|  | Australia | 28.8% | 10.0% | 2.83 |
| E | Germany | 85.4% | 65.4% | 6.2 |
|  | Ecuador | 51.6% | 16.4% | 3.94 |
|  | Ivory Coast | 42.6% | 12.0% | 3.98 |
|  | Curacao | 20.4% | 6.2% | 2.65 |
| F | Netherlands | 89.4% | 64.4% | 6.88 |
|  | Sweden | 71.6% | 22.6% | 5.56 |
|  | Japan | 34.0% | 11.8% | 3.42 |
|  | Tunisia | 5.0% | 1.2% | 1.31 |
| G | Belgium | 82.2% | 53.8% | 5.71 |
|  | New Zealand | 60.6% | 28.4% | 5.23 |
|  | Iran | 39.6% | 14.2% | 3.73 |
|  | Egypt | 17.6% | 3.6% | 2.08 |
| H | Spain | 84.6% | 50.0% | 6.0 |
|  | Uruguay | 78.4% | 40.4% | 6.39 |
|  | Cape Verde | 32.2% | 8.4% | 3.61 |
|  | Saudi Arabia | 4.8% | 1.2% | 1.11 |
| I | Senegal | 75.6% | 39.4% | 5.48 |
|  | France | 74.4% | 42.8% | 5.03 |
|  | Iraq | 27.8% | 10.8% | 3.2 |
|  | Norway | 22.2% | 7.0% | 3.01 |
| J | Argentina | 91.4% | 75.2% | 6.68 |
|  | Algeria | 46.6% | 8.4% | 3.52 |
|  | Austria | 31.0% | 8.0% | 3.2 |
|  | Jordan | 31.0% | 8.4% | 3.4 |
| K | Colombia | 81.6% | 53.0% | 6.64 |
|  | Portugal | 71.6% | 32.4% | 4.9 |
|  | Congo DR | 24.0% | 8.4% | 2.63 |
|  | Uzbekistan | 22.8% | 6.2% | 2.77 |
| L | England | 84.8% | 51.8% | 5.9 |
|  | Croatia | 77.0% | 34.6% | 5.89 |
|  | Ghana | 34.6% | 12.6% | 4.29 |
|  | Panama | 3.6% | 1.0% | 1.14 |

## Predicted Round of 16 Qualifiers

| Group | 1st Place | 2nd Place |
|-------|-----------|-----------|
| A | Mexico | South Korea |
| B | Switzerland | Bosnia and Herzegovina |
| C | Brazil | Morocco |
| D | United States | Turkey |
| E | Germany | Ecuador |
| F | Netherlands | Sweden |
| G | Belgium | New Zealand |
| H | Spain | Uruguay |
| I | Senegal | France |
| J | Argentina | Algeria |
| K | Colombia | Portugal |
| L | England | Croatia |

---

## Quick Start

```bash
pip install -r requirements.txt
python cli.py train          # Train model + evaluate + simulate
python cli.py fixtures       # Show all 48 match predictions
python cli.py live results.json  # Update with real scores
python cli.py predict Brazil Argentina  # Single match
streamlit run dashboard.py   # Web UI
```

## Project Structure

```
fifa-predictor/
  cli.py                    # CLI: train, predict, simulate, live
  dashboard.py              # Streamlit web dashboard
  src/
    pipeline.py             # End-to-end FIFAPredictor
    data/loader.py          # 303 historical matches (1930-2022)
    features/builder.py     # 25 features per match
    models/
      ml_models.py          # XGBoost, LightGBM, RF, Neural Net
      statistical.py        # Poisson, Elo, Monte Carlo
      ensemble.py           # Stacked ensemble
    simulation/simulator.py # Group stage + live updates
    evaluation/metrics.py   # Accuracy, Brier, ROC-AUC
  config/config.yaml        # Hyperparameters
```

## How It Works

- **Poisson regression** models goal scoring as a Poisson process with attack/defense parameters
- **Elo ratings** capture head-to-head team strength dynamics
- **ML models** (XGBoost, LightGBM, RF, NN) use 25 engineered features including rolling form, ranking differentials, head-to-head history, and tournament experience
- **Stacked ensemble** combines all models via logistic regression meta-learner
- **Monte Carlo simulation** runs thousands of group stage scenarios to compute qualification probabilities
- **Live mode** locks in actual results and re-simulates remaining matches