#!/usr/bin/env python
import os
from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from knn import KNN

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

# our code
from utils import handle, main, weekdays, months, today

# cleans and saves daily breakdown of specific item sales to cleaned_day_item_array.csv
@handle("clean")
def clean():
    # load raw data
    dataset = pd.read_csv(r'C:\Users\bfeen\OneDrive\Desktop\coding_projects\JJ_ML_project\data\items-2022-03-03-2023-01-31.csv', low_memory=False)
    
    # get columns that're being used for the model
    df_raw = pd.DataFrame(dataset, columns = ['Date', 'Item', 'Qty'])
    
    # ignore "voided" item labels
    df_non_void = df_raw[~df_raw['Item'].str.contains('void', case=False)]

    # set data-selection mask - currently only selecting two-rivers products
    get = ((df_non_void['Item'].str.contains('sandwich', case=False))
    | (df_non_void['Item'].str.contains('bagel', case=False)) 
    | (df_non_void['Item'].str.contains('wrap', case=False)))

    # get desired data using mask
    df = df_non_void[get]

    list_days = df['Date'].unique()
    num_days = list_days.shape[0]

    list_items = df['Item'].unique()
    num_items = list_items.shape[0]
    print("Cleaning and exporting daily sales data for the following items...")
    print(list_items)

    # create new, faster, summarized dataset: row is day, column is item
    raw_cleaned = np.zeros(num_days * num_items).reshape(num_days, num_items)
    
    for i in range(num_days):
        for j in range(num_items):
            sift = df[(df['Item'] == list_items[j]) & (df['Date'] == list_days[i])]
            raw_cleaned[i, j] = sift.shape[0]

    cleaned_df = pd.DataFrame(raw_cleaned, columns=list_items)
    cleaned_df.insert(0, 'Date', list_days)
    fname = Path("..", "data", "cleaned_day_item_array.csv")
    cleaned_df.to_csv(fname)

    assert(num_items != None)

# helper; loads data into global vars from cleaned_day_item_array.csv
def load_data():
    fname = Path("..", "data", "cleaned_day_item_array.csv")
    data_all = pd.read_csv(fname)
    df_all = pd.DataFrame(data_all)
    return df_all

# helper; slices df_all and returns learnable/clean data for item param
# df_all must be initialized in order to call this
def get_cleaned_data(item, df_all):
        data = df_all.loc[:, ['Date', item]]
        n = data.shape[0]
        data_vec = np.array(data)
        
        weekday_arr = weekdays(data_vec[:, 0]).reshape(n, 1)
        weekday_arr = weekday_arr * 2
        month_arr = months(data_vec[:, 0]).reshape(n, 1)
        month_arr = month_arr * 3

        quantities_obj = np.array(data_vec[:, 1])
        quantities = np.round_(quantities_obj.astype(np.float32)).reshape(n, 1)

        date_data = np.append(month_arr, weekday_arr, axis = 1)
        
        ret = np.append(date_data, quantities, axis = 1)
        return ret

# helper that runs cross validation (with specified number of folds) on a KNN model with each k in ks, returning
#    an array cv_accs, where cv_accs[i] is the mean cross-validation error across all folds for ks[i]
def test(X, y, ks, folds):

    n = int(X.shape[0])
    num_folds = folds

    # this assertion ensures that your validation sets will always have at least one item in them
    assert(num_folds <= n)

    fold_size = int(n / num_folds)
    leftovers = int(n % num_folds)

    cv_accs = np.zeros(len(ks))
    for k in ks:
        # clear errors for this iteration
        errors = np.empty(num_folds, dtype=float)
        for i in range(num_folds):
            # clear mask
            mask = np.zeros(n, dtype=bool)

            # set mask for this iteration
            i_0 = (i * fold_size)
            i_1 = ((i + 1) * fold_size)
            # in case n is not evenly divisible by the number of fold we are doing,
            #     append the leftover data to the end of the last validation set
            if i == num_folds - 1:
                i_1 = i_1 + leftovers
            mask[i_0:i_1] = True

            # get X_validate and X_train from X using mask
            X_validate = X[mask, :]
            y_validate = y[mask]
            X_train = X[~mask, :]
            y_train = y[~mask]

            # train and validate on this fold, and store validation error in errors
            model = KNeighborsClassifier(n_neighbors= k)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_validate)
            err_abs = np.abs(y_pred - y_validate)
            err_rel_approx = err_abs / (y_validate + 1)
            
            err = np.mean(err_rel_approx)
            errors[i] = err
        
        # store average error for k across the 10 folds
        cv_accs[ks.index(k)] = np.mean(errors)
    return cv_accs

@handle("predict")
def predict():
    df_all = load_data()
    list_items = df_all.columns[2:]
    num_items = list_items.shape[0]

    print("Predicting today's sales for the following items...")
    print(list(list_items))
    num_items = list_items.shape[0]

    pred = np.zeros(num_items)

    ks = range(20, 31, 1)
    ensemble_size = len(ks)

    for item in list_items:

        data = get_cleaned_data(item, df_all)

        X = data[:, 0:2]
        y = data[:, 2]
        
        item_preds = np.zeros(ensemble_size)

        i = 0
        for k in ks:
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X, y)
            today_ = today().reshape(1, -1)
            item_preds[i] = model.predict(today_)
            i = i + 1
            
        pred[np.where(list_items == item)] = np.round(np.mean(item_preds))
    
    print(pred)

@handle('test_granular')
def test_granular():
    df_all = load_data()
    list_items = df_all.columns[2:]
    num_items = list_items.shape[0]

    print("Running granular cross-validation for the following items...")
    print(list(list_items))

    for item in list_items:

        data = get_cleaned_data(item, df_all)

        X = data[:, 0:2]
        y = data[:, 2]

        ks = range(1, 101, 1)
        folds = 20
        cv_accs = test(X, y, ks, folds)

        plt.clf()
        plt.plot(ks, cv_accs, label = '_err_cv')
        plt.xlabel('k')
        plt.ylabel('accuracy')
        plt.legend(loc = 'upper right')
        fname = Path("..", "figs", "granular", item.lower().replace(' ', '').replace(',', '_') + "_cv_accs.pdf")
        plt.savefig(fname)
        print(f"figure saved as {fname}")

@handle('test_aggregate')
def test_aggregate():
    df_all = load_data()
    list_items = df_all.columns[2:]
    num_items = list_items.shape[0]

    ks = range(1, 50, 1)
    folds = 10

    print("Running aggregate cross-validation for the following items...")
    print(list(list_items))
    print("Testing k's in: ")
    print(ks)
    print("Folds: ")
    print(folds)

    itemized_errors = np.zeros((num_items, len(ks)))

    for item in list_items:

        data = get_cleaned_data(item, df_all)

        X = data[:, 0:2]
        y = data[:, 2]

        itemized_errors[np.where(list_items == item)] = test(X, y, ks, folds, item)

    aggregate_cv_accs = np.mean(itemized_errors, axis = 0)
    
    plt.plot(ks, aggregate_cv_accs, label = 'item')
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.legend(loc = 'upper right')
    fname = Path("..", "figs", "aggregate_cv_accs.pdf")
    plt.savefig(fname)
    print(f"figure saved as {fname}")

@handle('granular_summary')
def test_aggregate():
    df_all = load_data()
    list_items = df_all.columns[2:]
    num_items = list_items.shape[0]

    ks = range(1, 50, 1)
    folds = 10

    print("Running aggregate cross-validation for the following items...")
    print(list(list_items))
    print("Testing k's in: ")
    print(ks)
    print("Folds: ")
    print(folds)

    for item in list_items:

        data = get_cleaned_data(item, df_all)

        X = data[:, 0:2]
        y = data[:, 2]

        ks = range(1, 101, 1)
        folds = 20
        cv_accs = test(X, y, ks, folds)
        plt.plot(ks, cv_accs, label = item.lower().replace(' ', '').replace(',', '_'))
    
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.legend(loc = 'upper right')
    fname = Path("..", "figs", "summary_granular_cv_accs.pdf")
    plt.savefig(fname)
    print(f"figure saved as {fname}")


if __name__ == "__main__":
    main()
