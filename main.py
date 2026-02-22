"""
Kevin Kuipers (s5051150)
Federico Berdugo Morales (s5363268)
Nik Skouf (s5617804)
"""

import random
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import error_analysis
import evaluation
import preprocessing

# fixed random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

MAX_ITERATIONS = 50000

LOGISTIC_REGRESSION_NAME = "Logistic regression"
SVM_NAME = "SVM"


def main() -> None:
    start_program_time = time.time()

    # Preprocessing

    starting_time = time.time()
    train, dev, test = preprocessing.preprocessing(RANDOM_SEED)
    preprocessing_time = time.time() - starting_time
    print(f"Preprocessing took {preprocessing_time}s")

    starting_time = time.time()
    train_x, train_y, dev_x, dev_y, test_x, test_y = preprocessing.tfidf_generator(
        train, dev, test
    )
    vector_time = time.time() - starting_time
    print(f"Generating vectors took {vector_time}s")

    # Training

    starting_time = time.time()

    logistic_regression = LogisticRegression(
        max_iter=MAX_ITERATIONS, random_state=RANDOM_SEED
    )

    logistic_regression.fit(train_x, train_y)
    dev_prediction_regression = logistic_regression.predict(dev_x)
    test_prediction_regression = logistic_regression.predict(test_x)

    regression_time = time.time() - starting_time
    print(f"{LOGISTIC_REGRESSION_NAME} took {regression_time}s")

    starting_time = time.time()
    svm = LinearSVC()

    svm.fit(train_x, train_y)

    dev_prediction_svm = svm.predict(dev_x)
    test_prediction_svm = svm.predict(test_x)
    svm_time = time.time() - starting_time
    print(f"{SVM_NAME} took {svm_time}s")

    # Evaluation

    evaluation.show_key_metrics(
        test_y, test_prediction_regression, LOGISTIC_REGRESSION_NAME
    )
    evaluation.show_key_metrics(test_y, test_prediction_svm, SVM_NAME)

    # Error analysis

    error_analysis.show_error_analysis(
        test_y, test_prediction_regression, LOGISTIC_REGRESSION_NAME
    )
    error_analysis.show_error_analysis(test_y, test_prediction_svm, SVM_NAME)

    end_program_time = time.time()
    duration_program = end_program_time - start_program_time
    print(f"Total program duration: {duration_program}")


if __name__ == "__main__":
    main()
