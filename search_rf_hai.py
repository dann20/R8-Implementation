import os
import itertools

import pandas as pd

from load_data import load_hai
from train_classifier import train_clf

def main():
    X_train, y_train, X_test, y_test = load_hai()

    params = {"n_estimators": [100, 250, 500, 1000],
              "max_features": [0.25, 0.3, 0.4, 0.5],
              "max_depth": [5, 10, 15],
              "min_samples_leaf": [10, 25, 50, 100],
              "oob_score": [True],
              "n_jobs": [-1]}

    keys, values = zip(*params.items())
    combs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    columns=[*list(params.keys()),
             'TrainAcc', 'TrainPrecision', 'TrainRecall', 'TrainF1',
             'TestAcc', 'TestPrecision', 'TestRecall', 'TestF1']

    df_result = pd.DataFrame(columns=columns)

    for param in combs:
        result_dict = train_clf('Random Forest', param, X_train, y_train, X_test, y_test)
        df_result = df_result.append(dict(param, **result_dict), ignore_index=True)

    outfile = './results/RF_search_hai.csv'
    df_result.to_csv(outfile, index=False, mode='a',
                     header=False if os.path.exists(outfile) else columns)

if __name__ == '__main__':
    main()
