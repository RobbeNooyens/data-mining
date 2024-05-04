import random

# Import packages
import pandas as pd
from sklearn import clone
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from Classfication.src.models import TrainingEnvironment
from Classfication.src.preprocessors import BiasPreprocessor, NumCatPreprocessor, ImputePreprocessor

# Set random seed to zero
random.seed(0)

# Load training data
df = pd.read_excel("assignment2_income.xlsx", "Data")

# Create preprocessors
numcat_preprocessor = NumCatPreprocessor(['age', 'workinghours'], ['workclass', 'education', 'marital status', 'occupation', 'ability to speak english'])
numcat_bias_preprocessor = NumCatPreprocessor(['age', 'workinghours'], ['workclass', 'education', 'marital status', 'ability to speak english'])
numcat_preprocessor_2 = NumCatPreprocessor(['age', 'workinghours', 'education', 'ability to speak english'], ['workclass', 'marital status', 'occupation'])
numcat_bias_preprocessor_2 = NumCatPreprocessor(['age', 'workinghours', 'education', 'ability to speak english'], ['workclass', 'marital status'])
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
    # Basic models
    'decision_tree_classifier': DecisionTreeClassifier(random_state=0),
    'knn_classifier': KNeighborsClassifier(),
    # Ensemble models
    'random_forest_classifier': RandomForestClassifier(random_state=0),
    'bagging_model': BaggingClassifier(estimator=DecisionTreeClassifier(random_state=0), random_state=0),
    'ada_boost': AdaBoostClassifier(estimator=DecisionTreeClassifier(random_state=0), random_state=0, algorithm="SAMME"),
    'gradient_boost': GradientBoostingClassifier(random_state=0),
}

# Train models with different preprocessors
env = TrainingEnvironment(df, 'income')
for name, classifier in models.copy().items():
    env.train(name + "_raw", clone(classifier), impute_preprocessor, numcat_preprocessor_2)
    env.tune(name + "_raw")
for name, classifier in models.copy().items():
    env.train(name + "_bias", clone(classifier), impute_preprocessor, basic_bias, numcat_preprocessor_2)
    env.tune(name + "_bias")
for name, classifier in models.copy().items():
    env.train(name + "_extended_bias", clone(classifier), impute_preprocessor, extended_bias, numcat_bias_preprocessor_2)
    env.tune(name + "_extended_bias")

# classifier = RandomForestClassifier(random_state=0),
# env.train("random_forest_classifier_bias", clone(classifier), impute_preprocessor, basic_bias, numcat_preprocessor_2)
# env.tune("random_forest_classifier_bias")

# Plot decision tree
env.plot_tree('decision_tree_classifier_bias')

# Save model performance
env.export("performance.xlsx")

# Load and run test data
df_test = pd.read_excel("assignment2_test.xlsx")
df_test['predicted income'] = env.predict('random_forest_classifier_bias_tuned', df_test)

# Save test data
df_test.to_excel("assignment2_test_predicted.xlsx", index=False)

