from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


LogisticRegression_params = {'C' : [0.1,1]}
DecisionTreeClassifier_params = {'ccp_alpha' : [0,1]}
RandomForestClassifier_params = {'ccp_alpha' : [0,1]}
GradientBoostingClassfier_params = {'max_depth' : [3,5],'subsample' : [0.5,1], 'max_features' : ['sqrt',None]}


def get_names_list():
    """Return list of model names

    Returns:
        [str]: [model name]
    """    
    return ['LogisticRegression',
            'DecisionTreeClassifier',
            'RandomForestClassifier',
            'GradientBoostingClassifier']

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
    return [LogisticRegression_params,
            DecisionTreeClassifier_params,
            RandomForestClassifier_params,
            GradientBoostingClassfier_params]