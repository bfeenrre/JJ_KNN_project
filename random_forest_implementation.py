import numpy as np
from sklearn.ensemble import RandomForestClassifier as random_forest
import data_interface
from utils import handle, main
import date_utils

items = data_interface.bakery_items + data_interface.food_items + data_interface.bottle_drink_items
df = data_interface.load_data()

@handle('test')
def test():
    print(f'testing a 100-tree random forest classifier trained with bootstrapped data on {len(items)} items...')

    num_tests = 5

    total_abs_error = 0
    total_rel_error = 0

    for item in items:
        item_total_abs_error = 0
        item_total_rel_error = 0
        # for each item....
        for test in range(0, num_tests, 1): 
            # 1. train a random forest classifier on augmented, bootstrapped, as I don't have much data to begin with), learnable data
            X, y = data_interface.get_learnable_data(df, item, bootstrap=True)
            model = random_forest(n_estimators=100, criterion="gini")
            model.fit(X, y)

            X_valid, y_valid = data_interface.get_learnable_data(df, item, bootstrap=True)
            y_pred = model.predict(X_valid)

            abs_residuals = y_valid - y_pred
            err_abs = np.mean(np.abs(abs_residuals))
            item_total_abs_error = item_total_abs_error + err_abs
            total_abs_error = total_abs_error + err_abs

            rel_residuals = abs_residuals
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_residuals[abs_residuals != 0] = np.true_divide(np.abs(abs_residuals), y_valid)[abs_residuals != 0]
                rel_residuals[rel_residuals == np.inf] = 1
                rel_residuals = np.nan_to_num(rel_residuals)
            err_rel = np.mean(rel_residuals)
            item_total_rel_error = item_total_rel_error + err_rel
            total_rel_error = total_rel_error + err_rel

        print(f'Absolute and relative errors across {num_tests} tests when predicting daily {item} sales:\n    {item_total_abs_error / num_tests}, {item_total_rel_error / num_tests}')

@handle('predict_today')
def predict_today():
    print(f'using a 100-tree random forest classifier trained with bootstrapped data on {len(items)} items to predict sales for today ({date_utils.today()}) ...')

    # for each item....
    for item in items: 
        # 1. train a random forest classifier on augmented, bootstrapped (x5, as I don't have much data to begin with), learnable data
        X, y = data_interface.get_learnable_data(df, item)
        model = random_forest(n_estimators=100, criterion="gini")
        model.fit(X, y)
        print(f'prediction for {item} for today: {model.predict(date_utils.today())}')

if (__name__ == "__main__"):
    main()