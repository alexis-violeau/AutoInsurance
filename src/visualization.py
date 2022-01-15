import seaborn as sns
import matplotlib.pyplot as plt


quant_columns = ['AGE', 'HOMEKIDS','YOJ', 'INCOME','HOME_VAL','TRAVTIME','KIDSDRIV', 'BLUEBOOK', 'TIF','OLDCLAIM', 'CLM_FREQ','MVR_PTS', 'CAR_AGE']
cat_columns = ['TARGET_AMT', 'MSTATUS', 'SEX', 'EDUCATION','JOB', 'CAR_USE', 'CAR_TYPE', 'RED_CAR', 'URBANICITY']
target = ['TARGET_FLAG']


def visualize_data(df):
    fig, ax = plt.subplots(nrows = len(quant_columns + cat_columns) + 1,figsize = (10,100))
    
    hue = 'TARGET_FLAG'

    for (i,col) in enumerate(quant_columns + cat_columns):
        sns.histplot(df, x = col, hue = hue, ax = ax[i], kde = (col in quant_columns),common_norm=False, stat='density')
        
    sns.histplot(df,x = 'TARGET_FLAG', ax = ax[-1])
    
    plt.figure()
    sns.heatmap(df[quant_columns].corr())
    
    plt.show()