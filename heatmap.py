import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model, Model

class HeatmapGenerator:
    def __init__(self, model_path, input_data, layer_indices):
        self.model_path = model_path
        self.input_data = input_data
        self.layer_indices = layer_indices
        self.model = None
        self.layer_names = None
        self.activation_model = None
        self.heatmap = None
        self.averaged_heatmap = None

    def load_model(self):
        self.model = load_model(self.model_path)
        self.layer_names = [layer.name for layer in self.model.layers[self.layer_indices]]
        self.activation_model = Model(inputs=self.model.input, outputs=[layer.output for layer in self.model.layers[self.layer_indices]])

    def compute_heatmap(self):
        n_samples = self.input_data.shape[0]
        heatmap_shape = (n_samples,) + self.activation_model.output_shape[-3:]
        self.heatmap = np.empty(heatmap_shape)

        for i in range(n_samples):
            img_tensor = np.expand_dims(self.input_data[i], axis=0)
            activations = self.activation_model.predict(img_tensor)
            self.heatmap[i] = activations[-1]

        self.averaged_heatmap = np.mean(self.heatmap, axis=(0, 3))

    def plot_heatmap(self):
        plt.matshow(self.averaged_heatmap, cmap='bwr')
        plt.show()
        plt.savefig("heatmap.png")


model_path = "/content/drive/MyDrive/Colab Notebooks/data/heatmap/output/cnn/AO_model_101.hdf5"
input_data = np.load("/content/drive/MyDrive/Colab Notebooks/data/hgt/test_set.npy")
layer_indices = [-1]

heatmap_model = HeatmapGenerator(model_path, input_data, layer_indices)
heatmap_model.load_model()
heatmap_model.compute_heatmap()
heatmap_model.plot_heatmap()