Tested Python version: 3.13.8

Required libraries: see requirements.txt

To train the model: python tune_xgb_timesplit.py --input data.csv --target y_closed --n-iter 1000

Note that the current best parameter are in y_closed_best_param.pkl, training a new model will rewrite it.

The training time varies depending on compute power, but should generally finish within 30 minutes.

See analysis details in case.ipynb