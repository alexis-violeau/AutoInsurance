from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import numpy as np

quant_features = ['AGE', 'INCOME','TRAVTIME','KIDSDRIV', 'BLUEBOOK', 'TIF','CLM_FREQ','CAR_AGE']
cat_features = ['MSTATUS', 'SEX', 'HIGHSCHOOL','HIGHSPC', 'CAR_USE', 'CAR_TYPE', 'RED_CAR', 'URBANICITY']
features = quant_features + cat_features
target = ['TARGET_FLAG']


def preprocess_data(df):
    """Apply data transformation to training and test set.

    Args:
        train (pandas.DataFrame): training set (should include "TARGET_FLAG" column)
        test (pandas.DataFrame): test set

    Returns:
        (numpy.array,numpy.array,numpy.array): (X_train_preprocess, y_train, X_test_preprocess)
    """    
    
    # We binarise EDUCATION and JOB column to reduce the number of dimension for one hot encoding
    df['HIGHSCHOOL'] = df.EDUCATION.isin(['Bachelors','Masters','PhD'])

    df['HIGHSPC'] = df.JOB.isin(['Doctor','Lawyer','Manager'])
    
    # We one hot encode categorical features and scale numerical features
    preprocessor = ColumnTransformer([("quant_columns",StandardScaler(),quant_features),
                                      ("cat_columns",OneHotEncoder(),cat_features)],
                                     remainder = 'drop')
    
    
    X_train,y_train,X_val,y_val = train_test_split(df[features],df[target],train_size = 0.7, stratify = df[target])
    
    X_train_preprocess = np.nan_to_num(preprocessor.fit_transform(X_train),0)
    X_val_preprocess = np.nan_to_num(preprocessor.transform(X_val),0)
    
    y_train = y_train.values.ravel()
    y_val = y_val.values.ravel()
    
    return X_train_preprocess, y_train, X_val_preprocess, y_val
    
    
    
    
    
    
    