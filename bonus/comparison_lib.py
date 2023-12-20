import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import confusion_matrix

#import grid_search & random forest
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from colorspacious import cspace_converter
plt.style.use('ggplot')
import datetime
import sys

def get_paths_and_names():
    #get all path from argv, then get all names after the '-n' flag
    csv_paths = []
    names = []
    if len(sys.argv) < 2:
        print("Please provide at least one csv path")
        exit()
    for i in range(1, len(sys.argv)):
        if sys.argv[i] == '-n':
            names = sys.argv[i+1:]
            break
        else:
            csv_paths.append(sys.argv[i])
    #if no names provided, return None      
    if len(names) == 0:
        names = None
    return csv_paths, names

def get_df_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    df.fillna(0, inplace=True)
    return df

def get_models():
    models = []
    #models append (model_name, best grid search estimator)
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    models.append(('SGD', SGDClassifier()))
    models.append(('MLP', MLPClassifier()))
    models.append(('RF', RandomForestClassifier()))
    return models

def get_params():
    params = {}
    #params append (model_name, parameters)
    params['LR'] = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    params['LDA'] = {'solver': ['svd', 'lsqr', 'eigen']}
    params['KNN'] = {'n_neighbors': [5, 7, 9, 11, 13, 15]}
    params['CART'] = {'criterion': ['gini', 'entropy']}
    params['NB'] = {}
    params['SVM'] = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    params['SGD'] = {'loss': ['hinge', 'log', 'squared_hinge'], 'penalty': ['l1', 'l2']}
    params['MLP'] = {'hidden_layer_sizes': [(10,10,10), (20,20,20), (30,30,30)], 'activation': ['tanh', 'relu'], 'solver': ['sgd', 'adam'], 'alpha': [0.0001, 0.05], 'learning_rate': ['constant','adaptive']}
    params['RF'] = {'n_estimators': [10, 100, 1000], 'max_features': ['sqrt', 'log2']}
    return params

def grid_search(df):
    #get the target column
    target = df.columns[-1]
    #get the features
    features = df.columns[:-1]
    #split the data
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    #get the models
    models = get_models()
    #get the parameters for each model
    params = get_params()
    #get the best estimator for each model
    best_estimators = []
    for name, model in models:
        #get the parameters for each model
        param_grid = params[name]
        #grid search
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose=3, n_jobs=-1)
        grid_result = grid.fit(X_train, y_train)
        #get the best estimator
        best_estimators.append((name, grid_result.best_estimator_))
    return best_estimators

def get_accuracy(model, csv_path):
    #get the dataframe
    df = get_df_from_csv(csv_path)
    #get the target column
    target = df.columns[-1]
    #get the features
    features = df.columns[:-1]
    #split the data
    X = df[features]
    y = df[target]
    #get the accuracy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)

def compare_models(csv_paths, names = None): #function to plot histogram of accuracies for each model for each datasets and then compare best accuracy of each dataset
    if names == None:
        names = []
        for i in range(len(csv_paths)):
            names.append("Dataset " + str(i+1))

    best_estimators = []
    for i in range(len(csv_paths)):
        df = get_df_from_csv(csv_paths[i])
        best_estimators.append(grid_search(df))
    
    best_accuracies_per_dataset = []
    best_models_per_dataset = []
    for i, model_estimators in enumerate(best_estimators):
        #get the accuracies for each model
        accuracies = []
        model_names = []
        best_accuracy = 0
        best_model = ""
        for name, model in model_estimators:
            accuracy = get_accuracy(model, csv_paths[i])
            accuracies.append(accuracy)
            model_names.append(name)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = name

        best_accuracies_per_dataset.append(best_accuracy)
        best_models_per_dataset.append(best_model)

        #plot the histogram

        plt.figure(figsize=(10, 5))
        plt.bar(model_names, accuracies)
        plt.title(names[i])
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.savefig("bonus/figures/" + names[i] + "_accuracy.png")
        plt.close()

    #plot the best accuracy for each dataset
    plt.figure(figsize=(10, 5))
    bars = plt.bar(names, best_accuracies_per_dataset)
    plt.title("Best Accuracy for each Dataset")
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    # write model name in the bar vertically
    for bar, model_name in zip(bars, best_models_per_dataset):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, model_name, 
                 verticalalignment='bottom', horizontalalignment='center', 
                 rotation='vertical', color='black', fontsize=8)
    plt.savefig("bonus/figures/best_accuracy.png")
    plt.close()






    


