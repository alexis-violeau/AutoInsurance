import loading
import preprocessing
import visualization
import evaluation
import interpretability

print('Loading Data')
df = loading.load_data()

print('Exploratory Data Analysis')
visualization.visualize_data(df = df)

print('Preprocessing Data')
X_train_preprocess, y_train, X_val_preprocess, y_val = preprocessing.preprocess_data(df = df)

print('Looking for best model')
best_estimator_name, best_estimator, best_score = evaluation.evaluate_model(data = (X_train_preprocess, y_train, X_val_preprocess, y_val))

print('Explain model')
interpretability.explain(estimator = best_estimator, estimator_type = best_estimator_name, data = X_val_preprocess)