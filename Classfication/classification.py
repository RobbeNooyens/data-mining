import random

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from Classfication.models import TrainingEnvironment
from Classfication.preprocessors import BiasPreprocessor, NumCatPreprocessor, ImputePreprocessor

# Set random seed to zero
random.seed(0)

df = pd.read_excel("assignment2_income.xlsx", "Sheet1")

# Preprocessors
numcat_preprocessor = NumCatPreprocessor(['age', 'workinghours'], ['workclass', 'education', 'marital status', 'occupation', 'ability to speak english'])
impute_preprocessor = ImputePreprocessor(fill_english=0, fill_gave_birth='No')
basic_bias = BiasPreprocessor(
    replace_husband_wife=True,
    drop_sex=True,
    drop_gave_birth=True,
)
extended_bias = BiasPreprocessor(
    replace_husband_wife=True,
    drop_sex=True,
    drop_gave_birth=True,
    drop_occupation=True
)

# Create models
models = {
    # Default models
    'decision_tree_classifier': DecisionTreeClassifier(),
    'knn_classifier': KNeighborsClassifier(),
    # 'nb_classifier': CategoricalNB(),
    # Advanced models
    'random_forest_classifier': RandomForestClassifier(n_estimators=100, max_depth=5),
    'bagging_model': BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, random_state=0),
    'ada_boost': AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2), n_estimators=100, random_state=0),
    'gradient_boost': GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0),
}

env = TrainingEnvironment(df, 'income')
# for name, classifier in models.copy().items():
#     env.train(name + "_raw", classifier, impute_preprocessor, numcat_preprocessor)
# for name, classifier in models.copy().items():
#     env.train(name + "_bias", classifier, impute_preprocessor, basic_bias, numcat_preprocessor)
for name, classifier in models.copy().items():
    env.train(name + "_extended_bias", classifier, impute_preprocessor, extended_bias, numcat_preprocessor)

# # Optionally plot decision tree
# env.plot_tree("decision_tree")


