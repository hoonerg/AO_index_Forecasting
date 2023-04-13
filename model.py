import numpy as np
import tensorflow as tf
from tensorflow import keras
from contextlib import redirect_stdout
from pathlib import Path


class Conv2DforAO:
    def __init__(self, main_dir, data_dir, output_dir):
        self.main_dir = Path(main_dir)
        self.data_dir = self.main_dir / "data" / data_dir
        self.output_dir = self.main_dir / "output" / "cnn" / output_dir
        self.callbacks_list = [
            keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=200),
            keras.callbacks.ModelCheckpoint(
                filepath=str(self.output_dir / 'AO_b_model_101.hdf5'),
                monitor='val_loss',
                save_best_only=True,
            )
        ]

    def load_data(self):
        train_x_path = self.data_dir / "hgt" / "train_set.npy"
        train_y_path = self.data_dir / "train_index.npy"
        valid_x_path = self.data_dir / "hgt" / "valid_set.npy"
        valid_y_path = self.data_dir / "valid_index.npy"
        test_x_path = self.data_dir / "hgt" / "test_set.npy"
        test_y_path = self.data_dir / "test_index.npy"

        self.train_x = np.load(train_x_path)
        self.train_y = np.load(train_y_path)
        self.valid_x = np.load(valid_x_path)
        self.valid_y = np.load(valid_y_path)
        self.test_x = np.load(test_x_path)
        self.test_y = np.load(test_y_path)

    def build_model(self):
        inputs = keras.Input(shape=self.train_x.shape[1:])
        conv1 = keras.layers.Conv2D(32, [5, 5], activation='linear', padding='same', strides=1)(inputs)
        pool1 = keras.layers.MaxPool2D((2, 2), strides=2, padding='same')(conv1)
        conv2 = keras.layers.Conv2D(64, [5, 5], activation='tanh', padding='same', strides=1)(pool1)
        pool2 = keras.layers.MaxPool2D((2, 2), strides=2, padding='same')(conv2)
        conv3 = keras.layers.Conv2D(128, [5, 5], activation='tanh', padding='same', strides=1)(pool2)
        flat = keras.layers.Flatten()(conv3)
        dense1 = keras.layers.Dense(50, activation='relu')(flat)
        outputs = keras.layers.Dense(1, activation=None)(dense1)

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005), loss='mse')

    def train_model(self, batch_size=557, epochs=700):
        self.history = self.model.fit(
            self.train_x, self.train_y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            callbacks=self.callbacks_list,
        )

    def save_model(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(str(self.output_dir / 'AO_model_101.hdf5'))

    def save_summary(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(str(self.output_dir / 'model_summary.md'), 'w') as f:
            with redirect_stdout(f):
                self.model.summary()

    def save_training_loss(self):
        self.output_dir.mkdir(parents=True,exist_ok=True)
        tr_loss = self.history.history['loss']
        tr_loss = np.array(tr_loss)
        tr_loss.astype('float32').tofile(str(self.output_dir / 'tr_loss.gdat'))

    def calculate_test_loss(self):
        model = keras.models.load_model(saved_model_path)
        test_loss = model.evaluate(test_x, test_y)
        print("Test loss:", test_loss)

    def run_model(self):
        AO_model = Conv2DforAO(main_dir, data_dir, output_dir)
        AO_model.load_data()
        AO_model.build_model()
        AO_model.train_model()
        AO_model.save_model()
        AO_model.save_summary()
        AO_model.save_training_loss()
        calculate_test_loss()        

if __name__ == "__main__":
    Conv2DforAO(main_dir="/content/drive/MyDrive/Colab Notebooks/", data_dir="data", output_dir="cnn").run_model()