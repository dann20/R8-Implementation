import os
import itertools

import pandas as pd

from load_data import load_hai
from train_classifier import train_clf

def main():
    X_train, y_train, X_test, y_test = load_hai()

    params = {"n_neighbors": [475, 500],
              "n_jobs": [-1]}

    keys, values = zip(*params.items())
    combs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    columns=[*list(params.keys()),
             'TrainAcc', 'TrainPrecision', 'TrainRecall', 'TrainF1',
             'TestAcc', 'TestPrecision', 'TestRecall', 'TestF1']

    df_result = pd.DataFrame(columns=columns)

    for param in combs:
        result_dict = train_clf('KNN', param, X_train, y_train, X_test, y_test)
        df_result = df_result.append(dict(param, **result_dict), ignore_index=True)

    outfile = './results/KNN_search_hai.csv'
    df_result.to_csv(outfile, index=False, mode='a',
                     header=False if os.path.exists(outfile) else columns)

if __name__ == '__main__':
    main()
