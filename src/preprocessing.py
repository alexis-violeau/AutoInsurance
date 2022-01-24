from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import numpy as np

quant_features = ['AGE', 'INCOME','TRAVTIME','KIDSDRIV', 'BLUEBOOK', 'TIF','CLM_FREQ','CAR_AGE']
cat_features = ['MSTATUS', 'SEX', 'HIGHSCHOOL','HIGHSPC', 'CAR_USE', 'CAR_TYPE', 'RED_CAR', 'URBANICITY']
features = quant_features + cat_features
target = ['TARGET_FLAG']


def preprocess_data(df_train,df_test):
    """Apply data transformation to training and test set.

    Args:
        df (pandas.DataFrame): training set (should include "TARGET_FLAG" column)

    Returns:
        np.array tuple, list: (X_train_preprocess, X_val_preprocess, y_train, y_val), features_names_preprocess
    """    
    
    # We binarise EDUCATION and JOB column to reduce the number of dimension for one hot encoding
    df_train['HIGHSCHOOL'] = df_train.EDUCATION.isin(['Bachelors','Masters','PhD'])
    df_test['HIGHSCHOOL'] = df_test.EDUCATION.isin(['Bachelors','Masters','PhD'])


    df_train['HIGHSPC'] = df_train.JOB.isin(['Doctor','Lawyer','Manager'])
    df_test['HIGHSPC'] = df_test.JOB.isin(['Doctor','Lawyer','Manager'])
    
    # Fill nan value by mean value for quantitative features
    df_train[quant_features] = df_train[quant_features].fillna(df_train[quant_features].mean())
    df_test[quant_features] = df_test[quant_features].fillna(df_test[quant_features].mean())

    # Fill nan value by mode value for categorical features
    df_train[cat_features] = df_train[cat_features].fillna(df_train[cat_features].mode())
    df_test[cat_features] = df_test[cat_features].fillna(df_test[cat_features].mode())

    
    # We one hot encode categorical features and scale numerical features
    preprocessor = ColumnTransformer([("quant_columns",StandardScaler(),quant_features),
                                      ("cat_columns",OneHotEncoder(),cat_features)],
                                     remainder = 'drop')
    
    
    
    X_train,X_val,y_train,y_val = train_test_split(df_train[features],df_train[target],train_size = 0.7, stratify = df_train[target])
    
    X_train_preprocess = np.nan_to_num(preprocessor.fit_transform(X_train),0)
    X_val_preprocess = np.nan_to_num(preprocessor.transform(X_val),0)
    
    y_train = y_train.values.ravel()
    y_val = y_val.values.ravel()
    
    features_names_preprocess = preprocessor.get_feature_names_out(X_train.columns)
    
    X_test_preprocess = np.nan_to_num(preprocessor.transform(df_test[features]),0)
    
    return (X_train_preprocess, X_val_preprocess, y_train, y_val), features_names_preprocess, X_test_preprocess
    
    
    
    
    
    
    