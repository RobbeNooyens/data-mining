from subprocess import call
from typing import Dict

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import export_graphviz, DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV

from preprocessors import Preprocessor


class TrainingEnvironment():

    def __init__(self, df: DataFrame, target: str):
        self.df = df
        self.target = target
        self.X = self.df.drop(self.target, axis=1)
        self.y = self.df[self.target]
        self.trained_models: Dict[str, Pipeline] = {}
        self.scores = []

    def train(self, name: str, classifier, *preprocessors: Preprocessor):
        if name in self.trained_models:
            print(f"Model {name} already trained")
            return self.trained_models[name]

        # Build pipeline
        pipeline = [(step.name(), step) for step in preprocessors]
        pipeline.append(('classifier', classifier))

        # Copy training data
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.1, random_state=0)

        # Create and train model
        model = Pipeline(steps=pipeline)
        model.fit(X_train, y_train)

        # Score model
        y_pred = model.predict(X_test)
        self.assess_results(name, X_test, y_test, y_pred)

        # Save model
        self.trained_models[name] = model
        return model

    def predict(self, model_name: str, X: DataFrame):
        # Run a saved model on a new dataset
        if not model_name in self.trained_models:
            print(f"Model {model_name} not found")
            return
        return self.trained_models[model_name].predict(X.copy())

    def plot_tree(self, model_name):
        if not model_name in self.trained_models:
            print(f"Model {model_name} not found")
            return
        # Get decision tree
        decision_tree = self.trained_models[model_name]
        feature_names_transformed = decision_tree.named_steps['preprocessor'].get_feature_names_out()
        classifier = decision_tree.named_steps['classifier']
        try:
            # Visualize the tree using these feature names
            export_graphviz(classifier, out_file=f'{model_name}.dot',
                            feature_names=feature_names_transformed,
                            class_names=[str(cls) for cls in classifier.classes_],
                            rounded=True, proportion=False,
                            precision=2, filled=True)
            # Convert to png using system command (requires Graphviz)
            call(['dot', '-Tpng', f'{model_name}.dot', '-o', f'{model_name}.png', '-Gdpi=100'])
        except Exception as e:
            print(f"Error plotting tree: {e}")

    def tune(self, model_name: str):
        # Tune hyperparameters of a saved model and save it as a new model
        if model_name not in self.trained_models:
            print(f"Model {model_name} not found")
            return

        # Retrieve the existing pipeline
        model_pipeline = self.trained_models[model_name]
        classifier = model_pipeline.named_steps['classifier']

        # Define the hyperparameter grid based on the classifier's type
        if isinstance(classifier, DecisionTreeClassifier):
            param_grid = {
                'classifier__max_depth': [None, 5, 10, 20],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4]
            }
        elif isinstance(classifier, KNeighborsClassifier):
            param_grid = {
                'classifier__n_neighbors': [3, 5, 7, 9],
                'classifier__weights': ['uniform', 'distance']
            }
        elif isinstance(classifier, RandomForestClassifier):
            param_grid = {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [None, 5, 10, 20],
                'classifier__min_samples_split': [2, 5, 10]
            }
        elif isinstance(classifier, BaggingClassifier):
            param_grid = {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__estimator__max_depth': [None, 5, 10]
            }
        elif isinstance(classifier, AdaBoostClassifier):
            param_grid = {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__learning_rate': [0.01, 0.1, 1]
            }
        elif isinstance(classifier, GradientBoostingClassifier):
            param_grid = {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__learning_rate': [0.01, 0.1, 1],
                'classifier__max_depth': [1, 3, 5]
            }
        else:
            print(f"No hyperparameters defined for classifier of type {type(classifier).__name__}")
            return

        # Custom scorer that uses MSE of wrongly classified instances per gender and error type
        def fp_fn_sum_with_ref(y_true, y_pred, reference_col):
            # Check consistency of data lengths
            assert len(y_true) == len(y_pred) == len(reference_col), "Mismatched lengths"

            fpm = sum((y_pred == "high") & (y_true == "low") & (reference_col == "Male"))
            fpf = sum((y_pred == "high") & (y_true == "low") & (reference_col == "Female"))
            fnm = sum((y_pred == "low") & (y_true == "high") & (reference_col == "Male"))
            fnf = sum((y_pred == "low") & (y_true == "high") & (reference_col == "Female"))

            # Return mean squared error of all four metrics
            return np.mean(np.array([fpm, fpf, fnm, fnf]) ** 2)

        def fairness_scorer(gender_col):
            def scorer(y_true, y_pred):
                # Retrieve reference values matching y_true indices
                indices = y_true.index
                ref_subset = gender_col.iloc[indices]
                return fp_fn_sum_with_ref(y_true, y_pred, ref_subset)

            return make_scorer(scorer, greater_is_better=False)

        fairness = fairness_scorer(self.X["sex"])

        # Use GridSearchCV to tune the hyperparameters
        grid_search = GridSearchCV(model_pipeline, param_grid, scoring=fairness, cv=3, n_jobs=-1)

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.1, random_state=0)

        grid_search.fit(X_train, y_train)

        # Retrieve the best model
        best_model = grid_search.best_estimator_

        # Save and return the tuned model
        self.trained_models[model_name + "_tuned"] = best_model

        # Score model
        y_pred = best_model.predict(X_test)
        self.assess_results(model_name + "_tuned", X_test, y_test, y_pred)

        return best_model

    def assess_results(self, model: str, X_test, y_test, y_pred):
        fpm = sum((y_pred == "high") & (y_test == "low") & (X_test["sex"] == "Male"))
        fpf = sum((y_pred == "high") & (y_test == "low") & (X_test["sex"] == "Female"))
        fnm = sum((y_pred == "low") & (y_test == "high") & (X_test["sex"] == "Male"))
        fnf = sum((y_pred == "low") & (y_test == "high") & (X_test["sex"] == "Female"))

        tpm = sum((y_pred == "high") & (y_test == "high") & (X_test["sex"] == "Male"))
        tpf = sum((y_pred == "high") & (y_test == "high") & (X_test["sex"] == "Female"))
        tnm = sum((y_pred == "low") & (y_test == "low") & (X_test["sex"] == "Male"))
        tnf = sum((y_pred == "low") & (y_test == "low") & (X_test["sex"] == "Female"))

        # Get totals
        males = sum(X_test["sex"] == "Male")
        females = sum(X_test["sex"] == "Female")

        # Rates
        tprm = tpm / (tpm + fnm)
        fprm = fpm / (fpm + tnm)
        tprf = tpf / (tpf + fnf)
        fprf = fpf / (fpf + tnf)

        accuracy = accuracy_score(y_test, y_pred)
        f_score = f1_score(y_test, y_pred, average='weighted')

        self.scores.append((model, accuracy, f_score, fpm/males, fnm/males, fpf/females, fnf/females, tprm, fprm, tprf, fprf))

        print(f"[{model}]" + " Accuracy: {:.2f} F-measure: {:.2f} ".format(accuracy, f_score))

    def export(self, filename: str):
        df = pd.DataFrame(self.scores, columns=['Model', 'Accuracy', 'F-measure', 'FP Males', 'FN Males', 'FP Females', 'FN Females', 'TPR Males', 'FPR Males', 'TPR Females', 'FPR Females'])
        df.to_excel(filename, index=False)
