import loading
import preprocessing
import visualization
import evaluation
import interpretability
import pandas as pd

print('Loading Data:')

df_train, df_test = loading.load_data()
print('done')

print('Preprocessing Data:')

training_data, features_names_preprocess, X_test_preprocess = preprocessing.preprocess_data(df_train,df_test)
print('done /n')

print('Looking for best model:')

best_estimator_name, best_estimator, best_score = evaluation.evaluate_model(data = training_data)
print('done')

print('Using best model to submit predictions:')

y_pred = df_test[['INDEX','TARGET_FLAG']].copy()
y_pred['TARGET_FLAG'] = best_estimator.predict(X_test_preprocess)
y_pred.to_csv('submission/submission.csv',index = False)
print('done')
