import sys

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from Train import TrainModel as Train
from Predict import PredictFuture as Predict
from WindowGenerator import WindowGenerator
from Baseline import Baseline


def train_model(model_name, csv_path, model_save_path, MAX_EPOCHS=20, window_size=30):
    model = Train(
        model_name,
        csv_path,
        model_save_path,
        tf,
        pd,
        WindowGenerator,
        Baseline,
        IPython,
        int(MAX_EPOCHS),
    )
    model.define_window(int(window_size))
    model.train_model()
    model.evaluate_model()
    model.compile_and_save()
    model.plot_model_validation()


def predict_future(model_save_path, csv_path, window_size=30):
    prediction = Predict(model_save_path, csv_path, tf, np, pd, int(window_size))
    results = prediction.make_predictions()
    print(*(result for result in results), sep="\n\n")


def main():
    if sys.argv[1] == "--train":
        if len(sys.argv) == 7:
            train_model(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
        elif len(sys.argv) == 6:
            train_model(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
        else:
            train_model(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == "--predict":
        if len(sys.argv) == 5:
            predict_future(sys.argv[2], sys.argv[3], sys.argv[4])
        else:
            predict_future(sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    if len(sys.argv) < 5 and sys.argv[1] == "--train":
        print(
            "Usage is: $python3 main.py --train <model-name> <relative-csv-path> <relative-save-path> <epochs> <window_size>."
        )
        exit(1)
    elif len(sys.argv) < 4 and sys.argv[1] == "--predict":
        print(
            "Usage is: $python3 main.py --predict <relative-model-path> <relative-csv-path> <window_size_of_model>."
        )
        exit(1)
    elif sys.argv[1] != "--predict" and sys.argv[1] != "--train":
        print(
            "Usage is either: $python3 main.py --train <model-name> <relative-csv-path> <relative-save-path> <epochs>.\n"
            + "Or: $python3 main.py --predict <relative-model-path> <relative-csv-path>."
        )
        exit(1)
    main()
    exit(0)
