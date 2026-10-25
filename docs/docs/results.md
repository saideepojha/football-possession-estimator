| Method | Game 1 acc | Game 2 acc | Game 1 flips | Game 2 flips |
|---|---|---|---|---|
| Nearest player | 0.862 | 0.866 | 1933 | 2072 |
| + 3 m threshold, hold last | 0.797 | 0.767 | 1318 | 1396 |
| + rolling median (3 s) | 0.835 | 0.815 | 378 | 358 |
| Viterbi HMM on distance gap | 0.887 | 0.894 | 953 | 952 |
| LightGBM (trained on the other game) | 0.958 | 0.966 | 1289 | 988 |
| LightGBM + Viterbi | 0.964 | 0.971 | 407 | 327 |

Labelled frames: game 1 = 78,386, game 2 = 80,655. 'Flips' = number of possession changes predicted (fewer = less flicker).
