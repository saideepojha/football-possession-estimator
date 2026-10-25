# Football Possession Estimator

Estimates which team has possession in every frame of a football match,
using player and ball tracking data.

## Why this project
Built while preparing for More Than a Hack 2027, whose Initial Challenge is to
estimate team possession per frame from automatic tracking data.

## Data
Metrica Sports public sample data: https://github.com/metrica-sports/sample-data
The data is **not** included in this repo (see *Getting the data*).

## Approach
1. **Baseline:** nearest player to the ball, with a distance threshold
2. **Edge cases:** ball in the air, loose ball, dead ball
3. **Temporal smoothing:** rolling median and HMM to remove flicker
4. **ML model:** LightGBM with engineered features (distances, ball speed, closing velocity), validated by match

## Results
| Method | Accuracy | Notes |
|---|---|---|
| Nearest player | TBD | Simple, noisy |
| + smoothing | TBD | |
| + LightGBM | TBD | |

## What I learned / where it fails
_To be filled in after error analysis._

## Project structure
```
├── README.md
├── requirements.txt
├── notebooks/
├── src/
├── docs/
└── data/            (git-ignored)
```

## Getting the data
1. Clone https://github.com/metrica-sports/sample-data
2. Copy the files into `data/`

## How to run
```
pip install -r requirements.txt
jupyter notebook
```

## Roadmap
- [ ] Load data and plot a frame on a pitch
- [ ] Nearest-player baseline
- [ ] Smoothing
- [ ] LightGBM model
- [ ] Error analysis

## License
MIT. Author: Saideep Ojha
