import pandas as pd

amount_col = ['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM']

def load_data(path = 'data/auto-insurance-fall-2017/'):
    """Load data from Auto Insurance Kaggle competition. If data not available, please unzip auto-insurance-fall.zip.

    Args:
        path (str, optional): Path to data. Defaults to 'data/auto-insurance-fall-2017/'.
        
    Return:
        dataset (pandas.Dataframe)
    """    
    df_train = pd.read_csv(path + 'train_auto.csv')
    df_test = pd.read_csv(path + 'test_auto.csv')

    for col in amount_col:
        df_train[col] = df_train[col].str[1:].str.replace(',','').astype(float)
        df_test[col] = df_test[col].str[1:].str.replace(',','').astype(float)

 
    return df_train, df_test