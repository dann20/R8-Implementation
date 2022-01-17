from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

CLASSIFIER = {'RANDOM FOREST': RandomForestClassifier,
              'DECISION TREE': DecisionTreeClassifier,
              'KNN': KNeighborsClassifier}

def train_clf(name: str, params: dict, X_train, y_train, X_test, y_test):

    print(f'TRAINING {name} {params}...')
    clf = CLASSIFIER[name.upper()](**params)

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
    for train_idx, val_idx in sss.split(X_train, y_train):
        train_set, val_set = X_train[train_idx], X_train[val_idx]
        train_lbl, val_lbl = y_train[train_idx], y_train[val_idx]
        clf.fit(train_set, train_lbl)
        acc = clf.score(val_set, val_lbl)
        print(f'Iter accuracy: {acc}')

    print('Done training.')

    print('Testing set results:')
    accuracy = clf.score(X_test, y_test)
    y_pred = clf.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-score: {f1}')
    test_dict = {'TestAcc': accuracy,
                 'TestPrecision': precision,
                 'TestRecall': recall,
                 'TestF1': f1}

    print('Training set results:')
    accuracy = clf.score(X_train, y_train)
    y_pred = clf.predict(X_train)
    precision, recall, f1, _ = precision_recall_fscore_support(y_train, y_pred, average='binary')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-score: {f1}')
    train_dict = {'TrainAcc': accuracy,
                  'TrainPrecision': precision,
                  'TrainRecall': recall,
                  'TrainF1': f1}

    clf_dict = dict(train_dict, **test_dict)
    print('------------------------------------------------------------------------')
    return clf_dict
