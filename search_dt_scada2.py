import os
import itertools

import pandas as pd

from load_data import load_scada2
from train_classifier import train_clf

def main():
    X_train, y_train, X_test, y_test = load_scada2()

    params = {"max_depth": [3, 5, 7, 9, 10],
              "min_samples_split": [4, 6, 8, 10, 15, 20],
              "min_samples_leaf": [2, 4, 6, 8]}

    keys, values = zip(*params.items())
    combs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    columns=[*list(params.keys()),
             'TrainAcc', 'TrainPrecision', 'TrainRecall', 'TrainF1',
             'TestAcc', 'TestPrecision', 'TestRecall', 'TestF1']

    df_result = pd.DataFrame(columns=columns)

    for param in combs:
        result_dict = train_clf('Decision Tree', param, X_train, y_train, X_test, y_test)
        df_result = df_result.append(dict(param, **result_dict), ignore_index=True)

    outfile = './results/DT_search_scada2.csv'
    df_result.to_csv(outfile, index=False, mode='a',
                     header=False if os.path.exists(outfile) else columns)

if __name__ == '__main__':
    main()
