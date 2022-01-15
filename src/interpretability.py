import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import plot_tree
import shap


def explain(estimator, estimator_type, feature_names, data = None):
    """Explain estimator according to its type

    Args:
        estimator (model): model instance to explain
        estimator_type (str): model type
        feature_names (list): list of feature names
        data (np.array, optional): input data used for some interpretability techniques. Defaults to None.
    """    
    if estimator_type == 'LogisticRegression':
        explain_logistic(estimator,feature_names)
    if estimator_type == 'DecisionTree':
        explain_tree(estimator,feature_names)
    if estimator_type == 'RandomForest':
        explain_random_forest(estimator,feature_names)
    if estimator_type == 'GradientBoosting':
        explain_gradient_boosting(estimator,feature_names,data)
    
    
    
def explain_logistic(estimator,feature_names):

    
    p = len(feature_names)

        
    plt.figure(figsize = (20,10))
    plt.scatter(range(p),estimator.coef_)
    plt.grid(True)
    plt.xticks(range(p),feature_names,rotation = 90)
    plt.show()  
        
    
    
def explain_tree(estimator,feature_names):

    plt.figure(figsize = (20,10))
    plot_tree(estimator,feature_names=feature_names,filled = True,proportion = True,max_depth = 3,fontsize = 7,impurity = False)
    plt.show()
    
    
    
def explain_random_forest(estimator,feature_names):
    
    p = len(feature_names)
    
    plt.figure(figsize = (20,10))
    plt.scatter(range(p),estimator.feature_importances_)
    plt.xticks(range(p),feature_names,rotation = 90)
    plt.grid(True)
    plt.show()



def explain_gradient_boosting(estimator,feature_names,data):
        
    explainer_shap = shap.TreeExplainer(estimator.best_estimator_)
    shap_values = explainer_shap.shap_values(data)
    
    shap.summary_plot(shap_values, data,feature_names = feature_names)