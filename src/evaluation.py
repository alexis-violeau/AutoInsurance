import seaborn as sns
import matplotlib.pyplot as plt
import model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

    
def evaluate_model(data):
    
    """Search best model and best parameters through cross validation

    Returns:
        (str, model, float): (model name, model instance, model mean cross validation score)
    """    
    (X_train_preprocess, X_val_preprocess, y_train, y_val) = data
    
    names, estimators, grids = model.get_names_list(), model.get_models_list(), model.get_params_list()
    best_score = 0
    best_estimator = None
    best_estimator_name = None
    
    for name, classifier, params in zip(names, estimators, grids):
        grid_search = GridSearchCV(classifier, param_grid=params, n_jobs=-1,scoring = 'f1')
        clf = grid_search.fit(X_train_preprocess, y_train)
        
        if clf.best_score_ > best_score:
            best_estimator_name = name
            best_estimator = clf.best_estimator_
            best_score = clf.best_score_
        
        score = clf.score(X_val_preprocess, y_val)
        print("{} score: {}".format(name, score))
        
    y_pred = best_estimator.predict(X_val_preprocess)
    sns.heatmap(confusion_matrix(y_val,y_pred,normalize = 'all'),annot = True,cmap = plt.cm.Blues)
    plt.show()
    
    return best_estimator_name, best_estimator, best_score