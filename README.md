## Time Series Forecasting - Recursive Neural Network

The following program implements a Recursive Neural Network using TensorFlow.

The program in its current state predicts a single day in the future. Additonaly predications can be obtained by feeding the result of the previous iteration into the next calculation _this is not implemented currently_.

NOTE: This is an experiemental project that is **not** intended for production use. Financial markets are volatile and involve significant risk. This tool was developed as an experiemental and educational project. It is **not** intended to be used to make actual financial market decisions. It also has **not** been tested for correctness.

Usage:

```bash
python main.py --train <model-name> <relative-csv-path> <relative-save-path> <epochs> <window_size>
```

or

```bash
python main.py --predict <relative-model-path> <relative-csv-path> <window_size_of_model>
```
