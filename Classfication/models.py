from subprocess import call
from typing import List, Dict

from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz

from Classfication.preprocessors import BiasPreprocessor, NumCatPreprocessor, Preprocessor


def combine(pipeline, classifier):
    # Create copy of pipeline
    result = pipeline.copy()
    result.append(('classifier', classifier))
    return result

class TrainingEnvironment():

    def __init__(self, df: DataFrame, target: str):
        self.df = df
        self.target = target
        self.X = self.df.drop(self.target, axis=1)
        self.y = self.df[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.05, random_state=0)
        self.trained_models: Dict[str, Pipeline] = {}

    def train(self, name: str, classifier, *preprocessors: Preprocessor):
        if name in self.trained_models:
            print(f"Model {name} already trained")
            return self.trained_models[name]
        # Build pipeline
        pipeline = [(step.name(), step) for step in preprocessors]
        pipeline.append(('classifier', classifier))

        # Create, train and score model
        model = Pipeline(steps=pipeline)
        model.fit(self.X_train, self.y_train)
        score = model.score(self.X_test, self.y_test)

        # Save model
        print(f"[{name}]" + " Model score: {:.2f}".format(score))
        self.trained_models[name] = model
        return model

    def plot_tree(self, model_name):
        if not model_name in self.trained_models:
            print(f"Model {model_name} not found")
            return
        decision_tree = self.trained_models[model_name]
        feature_names_transformed = decision_tree.named_steps['preprocessor'].get_feature_names_out()
        classifier = decision_tree.named_steps['classifier']
        # Visualize the tree using these feature names
        export_graphviz(classifier, out_file='tree.dot',
                        feature_names=feature_names_transformed,
                        class_names=[str(cls) for cls in classifier.classes_],
                        rounded=True, proportion=False,
                        precision=2, filled=True)
        # Convert to png using system command (requires Graphviz)
        call(['dot', '-Tpng', 'tree.dot', '-o', f'{model_name}.png', '-Gdpi=100'])
