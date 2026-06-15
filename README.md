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
| Mexico vs United States | 37.0% | 22.2% | 40.9% | **A** |
| Mexico vs Canada | 71.0% | 12.8% | 16.2% | **H** |
| Mexico vs New Zealand | 37.3% | 28.8% | 33.9% | **H** |
| United States vs Canada | 72.0% | 12.4% | 15.7% | **H** |
| United States vs New Zealand | 38.2% | 29.1% | 32.7% | **H** |
| Canada vs New Zealand | 19.9% | 21.3% | 58.8% | **A** |
### Group B

| Match | H Win | Draw | A Win | Prediction |
|-------|-------|------|-------|------------|
| England vs Iran | 69.5% | 15.1% | 15.4% | **H** |
| England vs Scotland | 62.1% | 18.5% | 19.4% | **H** |
| England vs Wales | 82.8% | 8.3% | 9.0% | **H** |
| Iran vs Scotland | 32.4% | 23.4% | 44.2% | **A** |
| Iran vs Wales | 59.0% | 20.7% | 20.2% | **H** |
| Scotland vs Wales | 64.8% | 18.3% | 17.0% | **H** |
### Group C

| Match | H Win | Draw | A Win | Prediction |
|-------|-------|------|-------|------------|
| Argentina vs Chile | 39.8% | 23.0% | 37.2% | **H** |
| Argentina vs Nigeria | 67.3% | 15.4% | 17.3% | **H** |
| Argentina vs South Korea | 64.9% | 15.2% | 19.9% | **H** |
| Chile vs Nigeria | 62.9% | 18.3% | 18.8% | **H** |
| Chile vs South Korea | 60.7% | 17.8% | 21.5% | **H** |
| Nigeria vs South Korea | 33.7% | 23.1% | 43.2% | **A** |
### Group D

| Match | H Win | Draw | A Win | Prediction |
|-------|-------|------|-------|------------|
| France vs Netherlands | 16.8% | 18.6% | 64.7% | **A** |
| France vs Senegal | 37.1% | 21.9% | 41.0% | **A** |
| France vs Ecuador | 42.3% | 24.5% | 33.2% | **H** |
| Netherlands vs Senegal | 68.6% | 14.6% | 16.8% | **H** |
| Netherlands vs Ecuador | 70.8% | 15.2% | 14.0% | **H** |
| Senegal vs Ecuador | 47.6% | 20.1% | 32.3% | **H** |
### Group E

| Match | H Win | Draw | A Win | Prediction |
|-------|-------|------|-------|------------|
| Spain vs Croatia | 39.4% | 20.6% | 40.0% | **A** |
| Spain vs Japan | 56.5% | 17.7% | 25.9% | **H** |
| Spain vs Costa Rica | 73.0% | 11.3% | 15.7% | **H** |
| Croatia vs Japan | 54.4% | 19.6% | 26.0% | **H** |
| Croatia vs Costa Rica | 70.3% | 13.2% | 16.6% | **H** |
| Japan vs Costa Rica | 53.4% | 19.1% | 27.5% | **H** |
### Group F

| Match | H Win | Draw | A Win | Prediction |
|-------|-------|------|-------|------------|
| Brazil vs Switzerland | 57.2% | 19.7% | 23.1% | **H** |
| Brazil vs Serbia | 78.1% | 11.0% | 10.8% | **H** |
| Brazil vs Cameroon | 85.2% | 7.8% | 7.0% | **H** |
| Switzerland vs Serbia | 61.9% | 17.0% | 21.0% | **H** |
| Switzerland vs Cameroon | 72.0% | 13.5% | 14.5% | **H** |
| Serbia vs Cameroon | 49.6% | 20.7% | 29.6% | **H** |
### Group G

| Match | H Win | Draw | A Win | Prediction |
|-------|-------|------|-------|------------|
| Germany vs Denmark | 66.7% | 16.1% | 17.2% | **H** |
| Germany vs Uruguay | 41.0% | 25.2% | 33.8% | **H** |
| Germany vs Saudi Arabia | 90.1% | 3.9% | 6.0% | **H** |
| Denmark vs Uruguay | 20.0% | 21.5% | 58.5% | **A** |
| Denmark vs Saudi Arabia | 72.2% | 13.5% | 14.3% | **H** |
| Uruguay vs Saudi Arabia | 87.2% | 6.5% | 6.2% | **H** |
### Group H

| Match | H Win | Draw | A Win | Prediction |
|-------|-------|------|-------|------------|
| Portugal vs Belgium | 52.1% | 20.3% | 27.6% | **H** |
| Portugal vs Ghana | 57.1% | 17.1% | 25.8% | **H** |
| Portugal vs Australia | 73.1% | 11.5% | 15.4% | **H** |
| Belgium vs Ghana | 42.3% | 20.7% | 37.0% | **H** |
| Belgium vs Australia | 57.4% | 17.0% | 25.6% | **H** |
| Ghana vs Australia | 54.9% | 18.1% | 27.0% | **H** |
### Group I

| Match | H Win | Draw | A Win | Prediction |
|-------|-------|------|-------|------------|
| Italy vs Poland | 58.8% | 20.1% | 21.1% | **H** |
| Italy vs Morocco | 37.1% | 26.5% | 36.5% | **H** |
| Italy vs Qatar | 72.4% | 13.6% | 14.0% | **H** |
| Poland vs Morocco | 21.3% | 23.7% | 54.9% | **A** |
| Poland vs Qatar | 50.8% | 25.5% | 23.7% | **H** |
| Morocco vs Qatar | 68.3% | 17.7% | 14.0% | **H** |
### Group J

| Match | H Win | Draw | A Win | Prediction |
|-------|-------|------|-------|------------|
| Colombia vs Sweden | 50.3% | 21.7% | 28.0% | **H** |
| Colombia vs Ivory Coast | 71.8% | 12.7% | 15.5% | **H** |
| Colombia vs United Arab Emirates | 75.9% | 11.1% | 13.0% | **H** |
| Sweden vs Ivory Coast | 59.2% | 18.2% | 22.6% | **H** |
| Sweden vs United Arab Emirates | 65.6% | 16.8% | 17.6% | **H** |
| Ivory Coast vs United Arab Emirates | 49.0% | 22.2% | 28.8% | **H** |
### Group K

| Match | H Win | Draw | A Win | Prediction |
|-------|-------|------|-------|------------|
| Czech Republic vs Peru | 49.8% | 25.7% | 24.5% | **H** |
| Czech Republic vs Algeria | 37.6% | 24.6% | 37.7% | **A** |
| Czech Republic vs Panama | 76.5% | 9.8% | 13.7% | **H** |
| Peru vs Algeria | 24.4% | 26.4% | 49.2% | **A** |
| Peru vs Panama | 59.0% | 19.7% | 21.3% | **H** |
| Algeria vs Panama | 76.5% | 10.0% | 13.5% | **H** |
### Group L

| Match | H Win | Draw | A Win | Prediction |
|-------|-------|------|-------|------------|
| Turkey vs Ukraine | 43.7% | 21.9% | 34.4% | **H** |
| Turkey vs Tunisia | 64.4% | 17.0% | 18.7% | **H** |
| Turkey vs Paraguay | 34.9% | 23.2% | 42.0% | **A** |
| Ukraine vs Tunisia | 61.6% | 17.6% | 20.9% | **H** |
| Ukraine vs Paraguay | 30.7% | 22.7% | 46.7% | **A** |
| Tunisia vs Paraguay | 16.6% | 16.0% | 67.4% | **A** |

## Group Stage Simulation

Monte Carlo simulation (500 runs) showing qualification probabilities for each team.

| Group | Team | Qualify % | Win Group % | Avg Points |
|-------|------|-----------|-------------|------------|
| A | Mexico | 70.4% | 42.2% | 5.05 |
|  | United States | 69.2% | 34.8% | 5.21 |
|  | New Zealand | 45.8% | 20.6% | 4.57 |
|  | Canada | 14.6% | 2.4% | 1.94 |
| B | England | 92.2% | 75.2% | 6.82 |
|  | Scotland | 52.4% | 16.6% | 4.51 |
|  | Iran | 45.6% | 6.8% | 3.63 |
|  | Wales | 9.8% | 1.4% | 1.96 |
| C | Argentina | 81.2% | 52.6% | 5.78 |
|  | Chile | 77.2% | 35.8% | 5.47 |
|  | South Korea | 22.6% | 5.6% | 3.05 |
|  | Nigeria | 19.0% | 6.0% | 2.55 |
| D | Netherlands | 90.6% | 67.4% | 6.69 |
|  | France | 45.4% | 15.0% | 3.48 |
|  | Senegal | 41.8% | 11.0% | 3.78 |
|  | Ecuador | 22.2% | 6.6% | 2.86 |
| E | Spain | 80.4% | 47.8% | 5.53 |
|  | Croatia | 73.4% | 37.6% | 5.46 |
|  | Japan | 33.2% | 10.4% | 3.64 |
|  | Costa Rica | 13.0% | 4.2% | 2.32 |
| F | Brazil | 94.0% | 76.2% | 7.09 |
|  | Switzerland | 76.6% | 18.6% | 5.14 |
|  | Serbia | 20.2% | 3.0% | 2.93 |
|  | Cameroon | 9.2% | 2.2% | 1.94 |
| G | Germany | 87.0% | 58.4% | 6.34 |
|  | Uruguay | 69.4% | 32.4% | 5.87 |
|  | Denmark | 39.2% | 8.2% | 3.8 |
|  | Saudi Arabia | 4.4% | 1.0% | 1.13 |
| H | Portugal | 84.0% | 60.0% | 5.85 |
|  | Belgium | 61.8% | 22.8% | 4.59 |
|  | Ghana | 41.8% | 14.0% | 4.05 |
|  | Australia | 12.4% | 3.2% | 2.43 |
| I | Italy | 80.8% | 53.2% | 5.69 |
|  | Morocco | 68.4% | 31.6% | 5.52 |
|  | Poland | 35.8% | 11.2% | 3.37 |
|  | Qatar | 15.0% | 4.0% | 2.16 |
| J | Colombia | 88.6% | 64.6% | 6.37 |
|  | Sweden | 76.0% | 26.8% | 5.24 |
|  | Ivory Coast | 24.2% | 5.6% | 3.08 |
|  | United Arab Emirates | 11.2% | 3.0% | 2.25 |
| K | Czech Republic | 78.2% | 46.0% | 5.36 |
|  | Algeria | 67.6% | 35.4% | 5.71 |
|  | Peru | 45.8% | 16.8% | 4.01 |
|  | Panama | 8.4% | 1.8% | 1.83 |
| L | Turkey | 70.4% | 42.8% | 4.99 |
|  | Ukraine | 58.4% | 24.2% | 4.56 |
|  | Paraguay | 58.0% | 29.8% | 5.22 |
|  | Tunisia | 13.2% | 3.2% | 2.08 |

## Predicted Round of 16 Qualifiers

| Group | 1st Place | 2nd Place |
|-------|-----------|-----------|
| A | Mexico | United States |
| B | England | Scotland |
| C | Argentina | Chile |
| D | Netherlands | France |
| E | Spain | Croatia |
| F | Brazil | Switzerland |
| G | Germany | Uruguay |
| H | Portugal | Belgium |
| I | Italy | Morocco |
| J | Colombia | Sweden |
| K | Czech Republic | Algeria |
| L | Turkey | Ukraine |

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