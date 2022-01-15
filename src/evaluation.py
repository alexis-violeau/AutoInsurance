import seaborn as sns
import matplotlib.pyplot as plt
import model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

    
def evaluate_model(data):
    (X_train,y_train,X_val,y_val) = data
    
    y_train = y_train.values.ravel()
    y_val = y_val.values.ravel()
        
    names, estimators, grids = model.get_name_list(), model.get_model_list(), model.get_grid_list()
    best_score = 0
    best_estimator = None
    best_estimator_name = None
    
    for name, classifier, params in zip(names, estimators, grids):
        grid_search = GridSearchCV(classifier, param_grid=params, n_jobs=-1,scoring = 'f1')
        clf = grid_search.fit(X_train, y_train)
        
        if clf.best_score_ > best_score:
            best_estimator_name = name
            best_estimator = clf.best_estimator_
            best_score = clf.best_score_
        
        score = clf.score(X_val, y_val)
        print("{} score: {}".format(name, score))
        
    y_pred = best_estimator.predict(X_val)
    sns.heatmap(confusion_matrix(y_val,y_pred,normalize = 'all'),annot = True,cmap = plt.cm.Blues)
    plt.show()
    
    return best_estimator_name, best_estimator, best_score