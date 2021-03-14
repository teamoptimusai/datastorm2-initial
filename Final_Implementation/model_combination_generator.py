models = ['LogisticRegression(max_iter = 1500)',
'DecisionTreeClassifier(criterion="gini", max_depth=15)',
'GradientBoostingClassifier()',
'AdaBoostClassifier()',
'RandomForestClassifier()',
'RandomForestClassifier(random_state=0, n_jobs=-1, n_estimators=10, class_weight="balanced")',
'RandomForestClassifier(random_state=0, n_jobs=-1, n_estimators=10, class_weight="balanced_subsample")',
'ExtraTreesClassifier()',
'KNeighborsClassifier()',
'GaussianNB()']

from itertools import product

def models_combination_generator():
    string = "\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier, AdaBoostClassifier,ExtraTreesClassifier\nfrom sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.svm import SVC\nfrom sklearn.naive_bayes import GaussianNB\n\nmodels_combinations = [ "

    for x in product(models, repeat= 3):
        string += "["
        for y in x:
            string += y
            string += ","
        string = string[:-1]
        string += "]"
        string += ","
    string = string[:-1]
    string += "]"

    with open('models_combinations.py', 'w') as py_file:
        py_file.write(string)