#!/usr/bin/env python
import os
from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from knn import KNN
import time
from data_custodian import DataCustodian

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

# our code
from utils import handle, main, weekdays, months, today


## GOALS FOR THIS PROJECT FOR THE FUTURE
#   - implement bagging with my ensembles to reduce variance of my data
#   - add more dimensions to my feature space
#       - distance to a reading break/school holiday
#       - distance to christmas
#       - distance to new years
#       - proximity to finals season
#   - quantitatively find the best (or at least a near-optimal) weighting scheme for my feature space somehow

#   - QUANTITATIVELY TEST AGAINST JJ'S GOOGLE SHEETS SYSTEM - TEST IF KNN IS REALLY AN IMPROVEMENT

# DEFINITIONS
class Status:
    quit = 0 
    timer = 0
    model_mode = "knn"
main_options = "options:\n    - '-q' does what you think it does\n    - '-t' timer on/off\n    - '-m' choose what type of model you want to use\n    - 'pred' predict sales for today\n    - coming soon...\n"
mode_options = "options:\n    - 'knn' single knn model\n    - 'nknn' ensemble knn model\n    - 'r' return to main menu\n"

# GLOBAL VARIABLE
data_mng = DataCustodian()
model = None

@handle("run")
def run():
    # init
    status = Status()

    data_mng.load_and_clean_data()
    
    valid_commands = ["-o", "-q", "-t", "-m", "pred"]
    
    print(main_options)
    while(status.quit == False):
        command = input().lower().replace(" ", "")
        if command in valid_commands:
            if command in valid_commands[0:4]:
                status = update_status(status, command)
            else:
                if status.timer == 1:
                    t_s = time.time()
                run_func(command)
                if status.timer == 1:
                    t_e = time.time()
                    print("run time: " + str(t_e - t_s))
        else:
             print("please enter a valid command [type '-o' for list of valid commands]\n")
    print("see ya!")

def update_status(status, command):
    valid_modes = ["knn", "nknn"]
    if (command == "-q"):
        return 1
    if (command == "-t"):
        status.timer ^= status.timer
    if (command == "-o"):
        print("\n" + main_options)
    if (command == "-m"):
        print("\ncurrent model mode is... " + status.model_mode)
        print(mode_options)
        while(True):
            mode_sel = input().lower().replace(" ", "")
            if (mode_sel == "q"):
                return status
            if mode_sel in valid_modes:
                status.model_mode = mode_sel
                return status
            else:
                print("\ninvalid command...")
                print(mode_options)
    return status

def run_func(function):
    if function == "predict":
        if model != None:
            predict()
        else:
            print("Error: model == None, please train your model before predicting")
    
def clean_and_save(items):
    df_non_void = data_mng.cleaned_data_df

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

@handle('plot_data')
def plot_data():
    data = get_preprocessed_data('Bagel, Breakfast Meat', load_data())
    X = data[:, 0:2]
    plt.scatter(X[:, 0], X[:, 1])
    plt.colorbar()
    plt.show()

# helper; slices df_all and returns learnable/clean data for item param
# df_all must be initialized in order to call this
def get_preprocessed_data(item, df_all):
    data = df_all.loc[:, ['Date', item]]
    n = data.shape[0]
    data_vec = np.array(data)
    
    weekday_arr = weekdays(data_vec[:, 0]).reshape(n, 1)/6
    weekday_arr = weekday_arr * 3
    month_arr = months(data_vec[:, 0]).reshape(n, 1)/12
    month_arr = month_arr

    quantities_obj = np.array(data_vec[:, 1])
    quantities = np.round_(quantities_obj.astype(np.float32)).reshape(n, 1)

    date_data = np.append(month_arr, weekday_arr, axis = 1)
    
    ret = np.append(date_data, quantities, axis = 1)
    ret = ret.astype(np.int64)
    return ret

# helper that runs cross validation (with specified number of folds) on a KNN model with each k in ks, returning
#    an array cv_accs, where cv_accs[i] is the mean cross-validation error across all folds for ks[i]
def test(X, y, ks, folds, err_type):

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
            model = KNN(k=k)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_validate)
            err = np.abs(y_pred - y_validate)
            if err_type == 'rel':
                err = err / (y_validate + 1)
            
            err = np.mean(err)
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

    pred_tot = np.zeros(num_items)

    ensemble_size = 15

    for j in range(ensemble_size):
        pred = np.zeros(num_items)

        ks = np.round(find_ks()).astype(int)

        for i in range(num_items):

            data = get_preprocessed_data(list_items[i], df_all)

            X = data[:, 0:2]
            y = np.round(data[:, 2])

            model = KNN(k=ks[i])
            model.fit(X, y)
            today_ = today().reshape(1, -1)
                
            pred[i] = model.predict(today_)
    
        pred_tot += pred
    print(np.round(pred_tot / ensemble_size))

@handle('test_granular')
def test_granular():
    df_all = load_data()
    list_items = df_all.columns[2:]
    num_items = list_items.shape[0]

    print("Running granular cross-validation for the following items...")
    print(list(list_items))

    for item in list_items:

        data = get_preprocessed_data(item, df_all)
        np.random.shuffle(data)

        X = data[:, 0:2]
        y = data[:, 2]

        ks = range(1, 21, 1)
        folds = 20
        cv_accs = test(X, y, ks, folds, 'abs')

        plt.clf()
        plt.plot(ks, cv_accs)
        plt.xlabel('k')
        plt.ylabel('err_abs')
        fname = Path("..", "figs", "granular", item.lower().replace(' ', '').replace(',', '_') + "_cv_accs.pdf")
        plt.savefig(fname)
        print(f"figure saved as {fname}")

@handle('test_aggregate')
def test_aggregate():
    df_all = load_data()
    list_items = df_all.columns[2:]
    num_items = list_items.shape[0]

    ks = range(1, 21, 1)
    folds = 20

    print("Running aggregate cross-validation for the following items...")
    print(list(list_items))
    print("Testing k's in: ")
    print(ks)
    print("Folds: ")
    print(folds)

    itemized_errors = np.zeros((num_items, len(ks)))

    for item in list_items:

        data = get_preprocessed_data(item, df_all)
        np.random.shuffle(data)

        X = data[:, 0:2]
        y = data[:, 2]

        itemized_errors[np.where(list_items == item)] = test(X, y, ks, folds, 'rel')

    aggregate_cv_accs = np.mean(itemized_errors, axis = 0)
    
    plt.plot(ks, aggregate_cv_accs, label = 'item')
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.legend(loc = 'upper right')
    fname = Path("..", "figs", "aggregate_cv_accs.pdf")
    plt.savefig(fname)
    print(f"figure saved as {fname}")

@handle('granular_summary')
def granular_summary():
    df_all = load_data()
    list_items = df_all.columns[2:]
    num_items = list_items.shape[0]

    ks = range(1, 21, 1)
    folds = 20

    print("Running aggregate cross-validation for the following items...")
    print(list(list_items))
    print("Testing k's in: ")
    print(ks)
    print("Folds: ")
    print(folds)

    for item in list_items:

        data = get_preprocessed_data(item, df_all)
        np.random.shuffle(data)

        X = data[:, 0:2]
        y = data[:, 2]

        cv_accs = test(X, y, ks, folds, 'rel')
        plt.plot(ks, cv_accs, label = item.lower().replace(' ', '').replace(',', '_'))
    
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.legend(loc = 'upper right')
    fname = Path("..", "figs", "summary_granular_cv_accs.pdf")
    plt.savefig(fname)
    print(f"figure saved as {fname}")

@handle('test_jj')
def test_jj():
    df_all = load_data()
    list_items = df_all.columns[2:]
    num_items = list_items.shape[0]

    print("Testing JJ's current model's accuracy for the following items: ")
    print(list_items)

    accs = np.zeros(num_items)
    j = 0
    for item in list_items:

        data = get_preprocessed_data(item, df_all)

        X = data[:, 0:2]
        y = data[:, 2]

        num_preds = y.shape[0] - 4
        y_pred = np.zeros(num_preds)
        for i in range(4, num_preds + 4, 1):
            y_pred[i - 4] = 0.4*(y[i-1]) + 0.3*(y[i-2]) + 0.2*(y[i-3]) + 0.1*(y[i-4])

        accs[j] = np.mean(np.abs((y[4:] - y_pred)/(y[4:] + 1)))
        j = j + 1

    print("Accuracies: ")
    print(accs)

def run_k_test(df_all):
    list_items = df_all.columns[2:]
    num_items = list_items.shape[0]

    best_ks = np.zeros(num_items)

    for i in range(num_items):

        data = get_preprocessed_data(list_items[i], df_all)
        np.random.shuffle(data)

        X = data[:, 0:2]
        y = data[:, 2]

        ks = range(1, 20, 1)
        folds = 25
        cv_accs = test(X, y, ks, folds, 'abs')
        best_ks[i]= ks[np.argmin(cv_accs)]
    return best_ks
        

@handle('find_best_ks')
def find_ks():
    df_all = load_data()
    list_items = df_all.columns[2:]
    num_items = list_items.shape[0]

    num_tests = 5

    #rint("Using cross-validation to find the best k for the following items...")
    #print(list(list_items))

    best_ks = np.zeros(num_items)

    for j in range(num_tests):
        best_ks += run_k_test(df_all)

    return best_ks/num_tests

@handle('test_best_knn')
def test_best_knn():
    df_all = load_data()
    list_items = df_all.columns[2:]
    num_items = list_items.shape[0]

    print("Testing my model's accuracy for the following items: ")
    print(list_items)

    ks_by_item = find_ks()

    accs = np.zeros(num_items)
    for j in range(num_items):


        data = get_preprocessed_data(list_items[j], df_all)
        np.random.shuffle(data)
        n = int(data.shape[0])

        X_train = data[:, 0:2]
        y_train = data[:, 2]
        test_indx = np.random.randint(n, size = int(n/2))
        test_data = data[test_indx]
        X_test = test_data[:, 0:2]
        y_test = test_data[:, 2]

        model = KNeighborsClassifier(n_neighbors=int(ks_by_item[j]))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accs[j] = np.mean(np.abs(((y_pred - y_test) + 1)/(y_pred + 1)))

    print("Accuracies: ")
    print(accs)

if __name__ == "__main__":
    main()