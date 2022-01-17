import json

from load_data import load_hai
from train_classifier import train_clf

def main():
    X_train, y_train, X_test, y_test = load_hai()

    knn_params = {"n_jobs": -1}
    dt_params = {}
    rf_params = {"n_jobs": -1}

    knn_result_dict = train_clf('KNN', knn_params, X_train, y_train, X_test, y_test)
    dt_result_dict = train_clf('Decision Tree', dt_params, X_train, y_train, X_test, y_test)
    rf_result_dict = train_clf('Random Forest', rf_params, X_train, y_train, X_test, y_test)

    result_dict = {'KNN': {'parameters': knn_params,
                           'results': knn_result_dict},
                   'DECISION TREE': {'parameters': dt_params,
                                     'results': dt_result_dict},
                   'RANDOM FOREST': {'parameters': rf_params,
                                     'results': rf_result_dict}}

    with open('./results/results_R8_HAI.json', 'a', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=4)

if __name__ == '__main__':
    main()
