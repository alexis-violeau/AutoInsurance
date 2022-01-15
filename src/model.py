from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


Logistic_params = {'C' : [0.1,1]}
DecisionTree_params = {'ccp_alpha' : [0,1]}
RandomForest_params = {'ccp_alpha' : [0,1]}
GradientBoosting_params = {'max_depth' : [3,5],'subsample' : [0.5,1], 'max_features' : ['sqrt',None]}


def get_names_list():
    """Return list of model names

    Returns:
        [str]: [model name]
    """    
    return ['Logistic',
            'DecisionTree',
            'RandomForest',
            'GradientBoosting']

def get_models_list():
    """Return list of model instances

    Returns:
        [sklearn model]: [model instance]
    """    
    return [LogisticRegression(class_weight = 'balanced'),
            DecisionTreeClassifier(class_weight = 'balanced'),
            RandomForestClassifier(n_estimators = 200, class_weight = 'balanced_subsample'),
            GradientBoostingClassifier(n_estimators = 200)]
    
def get_params_list():
    """Return list of model parameters for grid search

    Returns:
        [dict]: [model parameters]
    """    
    return [Logistic_params,
            DecisionTree_params,
            RandomForest_params,
            GradientBoosting_params]