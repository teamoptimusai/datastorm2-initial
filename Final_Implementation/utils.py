import pandas as pd
from sklearn.utils import shuffle
import numpy as np
from sklearn.model_selection import train_test_split


def prepare_training_data(data_train, scaler, pca, validation=True):
    dates = ['Expected_checkin', 'Expected_checkout', 'Booking_date']
    one_hot_encoded_lst = ['Ethnicity', 'Educational_Level',
                           'Income', 'Country_region', 'Hotel_Type',
                           'Meal_Type', 'Deposit_type', 'Booking_channel']

    data_train = pd.get_dummies(data_train, columns=one_hot_encoded_lst)

    data_train['Gender'] = data_train['Gender'].map({'F': 0, 'M': 1})
    data_train['Visted_Previously'] = data_train['Visted_Previously'].map({
                                                                          'No': 0, 'Yes': 1})
    data_train['Previous_Cancellations'] = data_train['Previous_Cancellations'].map({
                                                                                    'No': 0, 'Yes': 1})
    data_train['Required_Car_Parking'] = data_train['Required_Car_Parking'].map({
                                                                                'Yes': 1, 'No': 0})
    data_train['Use_Promotion'] = data_train['Use_Promotion'].map(
        {'Yes': 1, 'No': 0})
    data_train['Reservation_Status'] = data_train['Reservation_Status'].map(
        {'Check-In': 0, 'Canceled': 1, 'No-Show': 2})

    checkin, canceled, noshow = np.bincount(data_train['Reservation_Status'])
    total = checkin + canceled + noshow

    weight_for_0 = (1 / checkin)*(total)/2.0
    weight_for_1 = (1 / canceled)*(total)/2.0
    weight_for_2 = (1 / noshow)*(total)/2.0

    class_weight = {0: weight_for_0, 1: weight_for_1, 2: weight_for_2}

    data_train[dates[0]] = pd.to_datetime(data_train[dates[0]])
    data_train[dates[1]] = pd.to_datetime(data_train[dates[1]])
    data_train[dates[2]] = pd.to_datetime(data_train[dates[2]])

    data_train['Expected_stay'] = (
        data_train[dates[1]] - data_train[dates[0]]).dt.days
    data_train['Booking_to_checkingin'] = (
        data_train[dates[0]] - data_train[dates[2]]).dt.days
    data_train['Month_of_stay'] = data_train[dates[0]].dt.month
    weekdayin = data_train[dates[0]].dt.dayofweek
    weekdayout = data_train[dates[1]].dt.dayofweek
    fina = []
    for x, y in zip(weekdayin, weekdayout):
        t = []
        if y >= x:
            for i in range(x, y + 1):
                t.append(i)
            if 5 in t or 6 in t:
                fina.append(1)
            else:
                fina.append(0)
        else:
            for i in range(x, 7):
                t.append(i)
            for j in range(0, y + 1):
                t.append(i)
            if 5 in t or 6 in t:
                fina.append(1)
            else:
                fina.append(0)
    data_train['weekend_stay'] = pd.DataFrame(fina, columns=['weekend_stay'])[
        'weekend_stay'].values
    data_train['Actual_cost'] = data_train['Expected_stay'] * \
        (data_train['Room_Rate']*(100 - data_train['Discount_Rate']))
    data_train = data_train.drop(dates, 1)

    for _ in range(1000):
        data_train = shuffle(data_train)

    cleaned_df = data_train.copy()

    eps = 0.001
    cleaned_df['Log Actual_cost'] = np.log(cleaned_df.pop('Actual_cost')+eps)
    cleaned_df['Log Room_Rate'] = np.log(cleaned_df.pop('Room_Rate')+eps)
    if validation:
        train1_df, train2_df = train_test_split(cleaned_df, test_size=0.2)
        train3_df, train1_df = train_test_split(train1_df, test_size=0.25)
        train3_df, val_df = train_test_split(train3_df, test_size=0.4)
    else:
        train1_df, train2_df = train_test_split(cleaned_df, test_size=0.2)
        train3_df, train1_df = train_test_split(train1_df, test_size=0.25)

    train1_labels = np.array(train1_df.pop('Reservation_Status'))
    train2_labels = np.array(train2_df.pop('Reservation_Status'))
    train3_labels = np.array(train3_df.pop('Reservation_Status'))
    if validation:
        val_labels = np.array(val_df.pop('Reservation_Status'))
        val_features = np.array(val_df)
    else:
        val_labels = None
        val_features = None

    train1_features = np.array(train1_df)
    train2_features = np.array(train2_df)
    train3_features = np.array(train3_df)

    if scaler:
        train1_features = scaler.fit_transform(train1_features)
        train2_features = scaler.transform(train2_features)
        train3_features = scaler.transform(train3_features)
        train1_features = np.clip(train1_features, -5, 5)
        train2_features = np.clip(train2_features, -5, 5)
        train3_features = np.clip(train3_features, -5, 5)
        if validation:
            val_features = scaler.transform(val_features)
            val_features = np.clip(val_features, -5, 5)

    if pca:
        pca.fit(train3_features)
        train1_features = pca.transform(train1_features)
        train2_features = pca.transform(train2_features)
        train3_features = pca.transform(train3_features)
        if validation:
            val_features = pca.transform(val_features)

    return train1_features, train1_labels, train2_features, train2_labels, train3_features, train3_labels, val_features, val_labels, class_weight


def prepare_data(df, scaler, pca):
    dates = ['Expected_checkin', 'Expected_checkout', 'Booking_date']
    one_hot_encoded_lst = ['Ethnicity', 'Educational_Level',
                           'Income', 'Country_region', 'Hotel_Type',
                           'Meal_Type', 'Deposit_type', 'Booking_channel']

    df = pd.get_dummies(df, columns=one_hot_encoded_lst)

    df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})
    df['Visted_Previously'] = df['Visted_Previously'].map({'No': 0, 'Yes': 1})
    df['Previous_Cancellations'] = df['Previous_Cancellations'].map({
                                                                    'No': 0, 'Yes': 1})
    df['Required_Car_Parking'] = df['Required_Car_Parking'].map(
        {'Yes': 1, 'No': 0})
    df['Use_Promotion'] = df['Use_Promotion'].map({'Yes': 1, 'No': 0})

    df[dates[0]] = pd.to_datetime(df[dates[0]])
    df[dates[1]] = pd.to_datetime(df[dates[1]])
    df[dates[2]] = pd.to_datetime(df[dates[2]])

    df['Expected_stay'] = (df[dates[1]] - df[dates[0]]).dt.days
    df['Booking_to_checkingin'] = (df[dates[0]] - df[dates[2]]).dt.days
    df['Month_of_stay'] = df[dates[0]].dt.month

    weekdayin = df[dates[0]].dt.dayofweek
    weekdayout = df[dates[1]].dt.dayofweek
    from pandas import DataFrame

    fina = []
    for x, y in zip(weekdayin, weekdayout):
        t = []
        if y >= x:
            for i in range(x, y + 1):
                t.append(i)
            if 5 in t or 6 in t:
                fina.append(1)
            else:
                fina.append(0)
        else:
            for i in range(x, 7):
                t.append(i)
            for j in range(0, y + 1):
                t.append(i)
            if 5 in t or 6 in t:
                fina.append(1)
            else:
                fina.append(0)
    df['weekend_stay'] = pd.DataFrame(fina, columns=['weekend_stay'])[
        'weekend_stay'].values
    df['Actual_cost'] = df['Expected_stay'] * \
        (df['Room_Rate']*(100 - df['Discount_Rate']))
    df = df.drop(dates, 1)

    eps = 0.001
    df['Log Actual_cost'] = np.log(df.pop('Actual_cost')+eps)
    df['Log Room_Rate'] = np.log(df.pop('Room_Rate')+eps)

    features = np.array(df)
    if scaler:
        features = scaler.transform(features)
        features = np.clip(features, -5, 5)
    if pca:
        features = pca.transform(features)

    return features
