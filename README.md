## Time Series Forecasting - Recursive Neural Network

The following program implements a Recursive Neural Network using TensorFlow.

The program in its current state predicts a single day in the future. Additonal predications can be obtained by feeding the result of the previous iteration into the next calculation _this is not implemented currently_.

NOTE: This is an educational example written as part of self-led studies regarding TensorFlow usage. Do **NOT** use this tool in production. It has not been tested nor is robust.

Usage:

```bash
python main.py --train <model-name> <relative-csv-path> <relative-save-path> <epochs> <window_size>
```

or

```bash
python main.py --predict <relative-model-path> <relative-csv-path> <window_size_of_model>
```
