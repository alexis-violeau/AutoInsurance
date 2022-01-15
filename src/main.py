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
data, features_names_preprocess = preprocessing.preprocess_data(df = df)

print('Looking for best model')
best_estimator_name, best_estimator, best_score = evaluation.evaluate_model(data = data)

print('Explain model')
interpretability.explain(estimator = best_estimator, estimator_type = best_estimator_name, feature_names = features_names_preprocess,data = data[0])