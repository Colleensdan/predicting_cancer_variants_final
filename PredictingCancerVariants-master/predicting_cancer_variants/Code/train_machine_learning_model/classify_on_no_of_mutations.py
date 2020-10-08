from __future__ import absolute_import, division, print_function, unicode_literals

import os
import pickle
import time

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from seaborn import heatmap
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from predicting_cancer_variants.Code.common.configuration import Configuration
from predicting_cancer_variants.Code.train_machine_learning_model.data_retriever import DataRetriever
from predicting_cancer_variants.Code.train_machine_learning_model.dummy_classifier import Never1Classifier


# noinspection PyPep8Naming
class MutationCardinalityTrainer:
    def __init__(self, cancer_ids=[], manually_test_model=False, show_confusion=True, cross_validate=True,
                 compare_with_dummy=True, precision_and_recall_verbosity=True, show_precision_and_recall=True):
        self._configuration = Configuration()
        self.__logger = self._configuration.get_logger(__name__)

        self.__retriever = DataRetriever()
        self.df = self.__retriever.fetch()

        self.cancer_ids = cancer_ids
        self.show_confusion = show_confusion
        self.cross_validate = cross_validate
        self.manually_test_model = manually_test_model
        self.compare_with_dummy = compare_with_dummy
        self.verbose_level_precision_and_recall = precision_and_recall_verbosity
        self.show_precision_and_recall_graph = show_precision_and_recall

    def train(self):
        """

        :return:
        """
        if self.df.empty:
            message = "Dataframe is empty. Cannot train on data of size 0"
            self.__logger.error(message)
        print(self.df['cType'].unique())

        # Assign columns to be categorised
        columns = ['Protein', 'Gene', "MtType", ("cType", "cName"), "studyId"]
        self.df = self.__create_categorical_col(self.df, columns)

        df = self.df.loc[self.df['cType'].isin(self.cancer_ids)]
        if not self.cancer_ids or len(self.cancer_ids) <= 1:
            message = "Incorrect number of cancer types. Cannot train for only", len(self.cancer_ids), "cancer types"
            self.__logger.error(message)
            return 0

        mut_per_patient, c_type = [], []
        for n, index in enumerate(self.cancer_ids):

            # patients with a certain cancer type
            patients_with_cancer_type = df.loc[df.cType == index, 'patient']

            # The number of mutations each patient has
            mutations_per_patient = patients_with_cancer_type.value_counts()
            mut_per_patient.append(mutations_per_patient)

            # Add n to cancer name to ensure cancer name unique per iteration
            c_type.append(np.full(mutations_per_patient.size, n))

        message = "Evaluated number of mutations per patient"
        self.__logger.info(message)

        # lambda function to flatten lists
        flatten = lambda l: [int(item) for sublist in l for item in sublist]
        mut_per_patient = flatten(mut_per_patient)
        c_type = np.concatenate(c_type)

        # transform python list to numpy array
        mut_per_patient = np.asarray(mut_per_patient)

        # clarify data and labels. X, y are commonly used to denote 'data' and 'labels' respectively
        X = mut_per_patient
        y = c_type

        X_train, X_test, y_train, y_test = self.__split_data_into_training_and_test(X, y, 0.2)

        # Ensures binary classification using bools for bivariate labelled data which improves performance
        if len(self.cancer_ids) == 2:
            y_train, y_test = self.__get_labels_for_binary_classifier(y_train, y_test)

        message = "Processed data and labels"
        self.__logger.info(message)

        # gets Pipeline which processes the data
        full_pipeline = self.__get_pipeline()
        message = "Pipeline established in memory"
        self.__logger.info(message)
        X_train_prepared = full_pipeline.fit_transform(X_train)
        message = "Training data successfully through the pipeline"
        self.__logger.info(message)

        # Creates classifier
        sgd_clf = SGDClassifier(random_state=42)
        message = "Starting classifier training"
        self.__logger.info(message)
        sgd_clf.fit(X_train_prepared, y_train)
        message = "Classifier training complete"
        self.__logger.info(message)

        # Prepare test data
        X_test_prepared = full_pipeline.fit_transform(X_train)
        message = "Test data successfully through the pipeline"
        self.__logger.info(message)

        # Show calculated precision and recall
        y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)
        f1_score = self.__precision_and_recall(y_train, y_train_pred, verbose=self.verbose_level_precision_and_recall)
        if f1_score:
            percentage = round(float(f1_score)*100, 2)
            print("----------------------------------------------------------------------------------------------------"
                  "-------------------------------------------------------------")
            message = f"Model trained successfully. Using F1 score as the performance metric, the model scored " \
                      f"{round(float(f1_score), 4)} which equates to {percentage}%"
            self.__logger.info(message)
            print("----------------------------------------------------------------------------------------------------"
                  "-------------------------------------------------------------")

        # Demonstrate ability to manually test input data
        if self.manually_test_model:
            test_data = (X_train_prepared[2]).todense()
            self.__test_model(test_data, sgd_clf, y_train[2])

        # Demonstrate ability to perform cross validation on model
        if self.cross_validate:
            self.__implement_cross_validation(X_test_prepared, y_test, sgd_clf)

        # Demonstrate effectiveness of model
        if self.compare_with_dummy:
            self.__compare_to_dummy_model(sgd_clf, X_train, y_train)

        # Confusion matrix
        if self.show_confusion:
            y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)
            self.__show_confusion_matrix(y_train, y_train_pred)

        # Determine which threshold to use
        if self.show_precision_and_recall_graph:
            self.__show_precision_recall_threshold_graph(sgd_clf, X_train, y_train)

        # Save model
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Models")
        if not os.path.exists(path):
            os.mkdir(path)

        # Finds actual cancer type from index
        cancers = ''
        for index in self.cancer_ids:
            var = (df.loc[df.cType == index, 'cName']).unique()
            cancers = cancers.join(var)

        # Create unique timestamp so all models are identifiable
        time_stamp = time.strftime("%Y%m%d-%H%M")
        file_name = cancers+"SGDClassifier"+time_stamp
        file_path = os.path.join(path, file_name + ".pkl")
        with open(file_path, "wb") as f:
            pickle.dump(sgd_clf, f)

    @staticmethod
    def __get_pipeline():
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        full_pipeline = FeatureUnion(
            transformer_list=[("num_pipeline", numeric_transformer),
                              ("cat_pipeline", categorical_transformer),
                              ]
        )

        return full_pipeline

    def __create_categorical_col(self, df, columns):
        """
        Transforms columns in a dataframe to categoric ints
        :param df: Dataframe to have columns changed
        :param columns: Columns to be categorised
                        OPTIONAL: tuple, (a,b) where a = column to be categorised and b = name of new column
                        the new column retains the information of the original column whilst the new column acts as the
                        integer unique classifier
        :return: dataframe containing classified columns
        """

        # Temporarily remove tuple such that columns can be checked
        for n, item in enumerate(columns):
            if isinstance(item, tuple):
                name, _ = item
                temporary_columns = columns.copy()
                temporary_columns[n] = name

        # Use appropriate var in validation
        if 'temporary_columns' in locals():
            column_set = temporary_columns
        else:
            column_set = columns


        for n, column in enumerate(columns):
            if type(column) == tuple:
                cat_col, new_col = column
                df[new_col] = df[cat_col]
                column = cat_col
            df[column], uniques = pd.factorize(df[column])
        return df

    @staticmethod
    def __get_labels_for_binary_classifier(y_train, y_test, true_val=0):
        y_train_x = (y_train == true_val)
        y_test_x = (y_test == true_val)
        return y_train_x, y_test_x

    @staticmethod
    def __split_data_into_training_and_test(X, y, test_size):
        # Splits data into training, and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        # Reshapes for classification
        X_train, x_test = np.reshape(X_train, (-1, 1)), np.reshape(X_test, (-1, 1))
        # Creates random index to shuffle 80% of the data
        arr_size = np.prod(X.shape)
        random_size = int(arr_size * 0.8)
        shuffle_index = np.random.permutation(random_size)
        X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
        return X_train, X_test, y_train, y_test

    def __test_model(self, test, model, expected):
        """
        Can test a model on a given input. Allows user to interact with model
        :param test: data to test model on. Must be of the same data type as trained the model with
        :param model: the trained model
        :param expected: correct outcome of test
        :return: message: prediction of model vs expected outcome
        """
        prediction = model.predict(test)
        message = "Prediction:", *prediction, "| Actual:", expected
        self.__logger.info(message)

    def __implement_cross_validation(self, X, y, model):
        """
        IMPLEMENTING CROSS VALIDATION
        Performs stratified sampling to produce folds that contain folds that contain
        a representative ratio of each class At each iteration the code creates a clone of the classifier,
        train that clone on the training folds and makes predictions on the test fold. Then it counts the number of
        correct predictions and outputs the ratio of correct predictions
        :param X: data to be fed to model
        :param y: labels for the data
        :param model: working model has been trained on X, and y
        :return: message. Returns ratio of correct predictions, which helps visualise performance metrics
        """

        skfolds = StratifiedKFold(n_splits=3, random_state=42)

        for train_index, test_index in skfolds.split(X, y):
            clone_clf = clone(model)
            X_train_folds = X[train_index]
            y_train_folds = y[train_index]
            X_test_fold = X[test_index]
            y_test_fold = y[test_index]

            clone_clf.fit(X_train_folds, y_train_folds)
            y_pred = clone_clf.predict(X_test_fold)
            n_correct = sum(y_pred == y_test_fold)
            message = "ratio of correct predictions: ", n_correct / len(y_pred)
            self.__logger.info(message)

    def __compare_to_dummy_model(self, model, X, y):
        # Measuring accuracy from trained model against dummy model
        trained_model = cross_val_score(model, X, y, cv=3, scoring="accuracy")
        message = "Accuracy from model:", trained_model
        self.__logger.info(message)
        never_1_classifier = Never1Classifier()
        dummy = cross_val_score(never_1_classifier, X, y, cv=3, scoring="accuracy")
        message = "Accuracy from dummy model:", dummy
        self.__logger.info(message)

    def __precision_and_recall(self, y_train, y_pred, average='micro', verbose = True):
        """
        Aquire precision and recall scores of a model based on classification success
        :param y_train: the correct labels
        :param y_pred: the labels the model predicts
        :param average: the average type to use when calculating precision and recall
        :return:
        """
        # Precision and recall
        pscore = precision_score(y_train, y_pred, average=average)
        rscore = recall_score(y_train, y_pred, average=average)

        # harmonic mean of precision
        f1 = f1_score(y_train, y_pred, average='micro')

        if verbose:
            message = F"Precision of model: {pscore} | Recall of model {rscore}"
            self.__logger.info(message)

            message = f"Harmonic Mean of precision and recall: F1 score: {f1}"
            self.__logger.info(message)
        return str(f1)

    @staticmethod
    def __show_confusion_matrix(y_train, y_pred):
        matrix = (confusion_matrix(y_train, y_pred))
        heatmap(matrix, annot=True)
        plt.show()

    @staticmethod
    def __show_precision_recall_threshold_graph(model, X, y):
        """
        Only callable if model is a binary classification
        :param model: trained model
        :param X: data model was trained with
        :param y: labels used to train
        :return: output graph
        """
        y_scores = cross_val_predict(model, X, y, cv=3, method='decision_function')
        precisions, recalls, thresholds = precision_recall_curve(y, y_scores)
        plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
        plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
        plt.xlabel("Threshold")
        plt.legend(loc="center left")
        plt.xlim([-800, 800])
        plt.ylim([0, 1])
        plt.show()


trainer = MutationCardinalityTrainer(cancer_ids=[5, 6])
trainer.train()
