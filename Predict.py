class PredictFuture:
    def __init__(self, model_save_path, csv_path, tf, np, pd, window_size=30):
        self.model_save_path = model_save_path
        self.csv_path = csv_path
        self.tf = tf
        self.np = np
        self.pd = pd
        self.model = tf.keras.models.load_model(f"{model_save_path}")
        self.data = pd.read_csv(self.csv_path)
        self.window_width = window_size
        self.data, self.train_mean, self.train_std = self.preprocess_data()
        self.define_window()

    def preprocess_data(self):
        data = self.data.drop(columns=["Date"])
        data = data[-self.window_width if -self.window_width > 0 else 1 :]
        train_mean = data.mean()
        train_std = data.std()
        data = (data - train_mean) / train_std
        return data, train_mean, train_std

    def define_window(self):
        window_width = self.window_width
        num_features = self.data.shape[1]
        num_data_points = len(self.data) - window_width + 1
        self.reshaped_data = self.np.zeros(
            (num_data_points, window_width, num_features)
        )
        for i in range(num_data_points):
            self.reshaped_data[i] = self.data[i : i + window_width]

    def make_predictions(self):
        predictions = self.model.predict(self.reshaped_data)
        original_format = (
            predictions * self.train_std["Close"] + self.train_mean["Close"]
        )
        return original_format
