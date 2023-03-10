from datetime import date
import numpy as np

# REQUIRES raw_dates is a single column dataframe with the column label 'Date' which contains date strings in an ISO format
def get_augmented_date_a(raw_dates):
    raw_dates['date_tuples'] = raw_dates['Date'].apply(str_to_quadruple)
    raw_dates['year'] = raw_dates['date_tuples'].map(lambda date : date[0])
    raw_dates['month'] = raw_dates['date_tuples'].map(lambda date : date[1])
    raw_dates['day_y'] = raw_dates['date_tuples'].map(lambda date : date[2])
    raw_dates['day_w'] = raw_dates['date_tuples'].map(lambda date : date[3])
    raw_dates['dist_xmas'] = raw_dates['day_y'].map(lambda date : np.min(np.abs(date - 359), np.abs(date + 6)))
    return raw_dates.loc[:, (raw_dates.columns != 'date_tuples') & (raw_dates.columns != 'Date')]

def str_to_quadruple(str):
    tuple = date.fromisoformat(str).timetuple()
    day_of_year = int(tuple.tm_yday)
    day_of_week = int(tuple.tm_wday)
    year, month, unused = (int(x) for x in str.split('-'))
    return (year, month, day_of_year, day_of_week)

def today():
    # generalize this later using datetime
    return np.asarray(str_to_quadruple(str(date.today()))).reshape(1, -1)
