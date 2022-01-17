import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

from utils import count

def load_scada2():
    print('LOADING RAW DATASET.....')
    df = pd.read_excel('./data/prior_value_minmax_scada2.xlsx', engine='openpyxl')

    print('PREPROCESSING.....')
    labels = df['binary result']
    drop_col = ['binary result', 'categorized result', 'specific result']
    df.drop(columns=drop_col, inplace=True)
    fs_drop_col = ['address', 'deadband', 'cycle time', 'control scheme'] # after feature selection, corr with target
    df.drop(columns=fs_drop_col, inplace=True)

    idx_split = -68658
    X_train = df[:idx_split]
    y_train = labels[:idx_split]
    X_test = df[idx_split:]
    y_test = labels[idx_split:]
    print(f'X_train: {count(y_train)}')
    print(f'X_test: {count(y_test)}')

    print('PERFORMING SMOTE.....')
    oversample = SMOTE()
    X_train, y_train = oversample.fit_resample(X_train, y_train)
    print(f'X_train: {count(y_train)} (after SMOTE)')

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()
    return X_train, y_train, X_test, y_test

def load_hai():
    print('LOADING RAW DATASET.....')
    df_train = [pd.read_csv(f'./data/train{i}.csv', sep=';') for i in range(1,3)]
    df_test = [pd.read_csv(f'./data/test{i}.csv', sep=';') for i in range(1,3)]
    trainset = pd.concat(df_train, ignore_index=True)
    testset = pd.concat(df_test, ignore_index=True)

    print('PREPROCESSING.....')

    select_cols = ["P1_B3005", "P4_ST_PS", "P2_VYT03", "P4_ST_PT01",
                   "P1_FCV03D", "P2_VXT03", "P1_B4022", "P1_B2004",
                   "P1_PIT01", "P1_B4002", "P1_FT01", "P1_B3004",
                   "P1_PCV02Z", "P1_LIT01", "P1_PCV02D", "P1_LCV01D",
                   "P2_SIT01", "attack"]

    trainset = trainset[select_cols]
    testset = testset[select_cols]

    train_labels = trainset['attack']
    test_labels = testset['attack']

    trainset.drop(columns=['attack'], inplace=True)
    testset.drop(columns=['attack'], inplace=True)

    print('NORMALIZING.....')
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(trainset)
    X_test = scaler.transform(testset)

    y_train = train_labels.to_numpy()
    y_test = test_labels.to_numpy()

    print(f'X_train: {count(y_train)}')
    print(f'X_test: {count(y_test)}')

    print('PERFORMING SMOTE.....')
    oversample = SMOTE()
    X_train, y_train = oversample.fit_resample(X_train, y_train)
    print(f'X_train: {count(y_train)} (after SMOTE)')

    return X_train, y_train, X_test, y_test
