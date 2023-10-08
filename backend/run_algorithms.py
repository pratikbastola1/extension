#!/usr/bin/python

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, metrics
from sklearn.metrics import classification_report, confusion_matrix


def calculate_metrics(y_test, Y_predicted):
    accuracy = metrics.accuracy_score(y_test, Y_predicted)
    print("accuracy = " + str(round(accuracy * 100, 2)) + "%")

    confusion_mat = confusion_matrix(y_test, Y_predicted)
    print(confusion_mat)

    print("TP\tFP\tFN\tTN\tSensitivity\tSpecificity")
    for i in range(confusion_mat.shape[0]):
        TP = confusion_mat[i, i]
        FP = confusion_mat[:, i].sum() - TP
        FN = confusion_mat[i, :].sum() - TP
        TN = confusion_mat.sum() - TP - FP - FN
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        print(f"{TP}\t{FP}\t{FN}\t{TN}\t{sensitivity:.2f}\t{specificity:.2f}")

    f_score = metrics.f1_score(y_test, Y_predicted, average='weighted')
    print(f_score)


def neural_network(dataset, class_labels, test_size):
    X = pd.read_csv(dataset)
    Y = pd.read_csv(class_labels)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

    model = MLPClassifier(hidden_layer_sizes=(100), activation='logistic', random_state=42)
    model.fit(X_train.values, y_train.values.ravel())
    Y_predicted = model.predict(X_test.values)

    return y_test, Y_predicted


def random_forests(dataset, class_labels, test_size):
    X = pd.read_csv(dataset)
    Y = pd.read_csv(class_labels)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

    model = RandomForestClassifier(n_estimators=5, criterion='entropy', random_state=42)
    model.fit(X_train.values, y_train.values.ravel())
    Y_predicted = model.predict(X_test.values)

    return y_test, Y_predicted


def support_vector_machines(dataset, class_labels, test_size):
    X = pd.read_csv(dataset)
    Y = pd.read_csv(class_labels)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

    model = svm.SVC(kernel='rbf', C=2.0)
    model.fit(X_train.values, y_train.values.ravel())
    Y_predicted = model.predict(X_test.values)

    return y_test, Y_predicted


def main():
    dataset = "Dataset.csv"
    class_labels = "Target_Labels.csv"
    test_size = 0.3

    print("\nrunning neural networks...")
    start_time = time.time()
    y_test, Y_predicted = neural_network(dataset, class_labels, test_size)
    calculate_metrics(y_test, Y_predicted)
    end_time = time.time()
    print("runtime = " + str(end_time - start_time) + " seconds")

    print("\nrunning random forests...")
    start_time = time.time()
    y_test, Y_predicted = random_forests(dataset, class_labels, test_size)
    calculate_metrics(y_test, Y_predicted)
    end_time = time.time()
    print("runtime = " + str(end_time - start_time) + " seconds")

    print("\nrunning support vector machines...")
    start_time = time.time()
    y_test, Y_predicted = support_vector_machines(dataset, class_labels, test_size)
    calculate_metrics(y_test, Y_predicted)
    end_time = time.time()
    print("runtime = " + str(end_time - start_time) + " seconds")


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print("Total runtime = " + str(end_time - start_time) + " seconds")
