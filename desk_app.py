from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np
import matplotlib.pyplot as plt

model = load_model('stamp_classification_model_resnet50.h5')

classes = joblib.load('classes.pkl')

label_encoder = LabelEncoder()
label_encoder.fit(classes)


def predict_stamp(img_path, model, label_encoder):
    img = load_img(img_path, target_size=(128, 128), color_mode='grayscale')
    img_array = img_to_array(img)
    img_array = np.repeat(img_array, 3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class)
    return predicted_label[0], img_array


def get_feature_map_image(model, img_array, layer_name, feature_index):
    layer = model.get_layer(name=layer_name)
    feature_model = Model(inputs=model.input, outputs=layer.output)
    feature_maps = feature_model.predict(img_array)
    feature_map = feature_maps[0, :, :, feature_index]
    plt.figure(figsize=(1.5, 1.5))
    plt.imshow(feature_map, cmap='viridis')
    plt.axis('off')
    plt.gcf().canvas.draw()
    img = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
    img = Image.fromarray(img)
    img = img.resize((150, 150), Image.Resampling.LANCZOS)

    plt.close()

    return img


class StampClassifierApp(object):

    def __init__(self):
        self.root = Tk()
        self.root.title("Stamp Classifier")
        self.root.resizable(0, 0)

        self.predict_button = Button(self.root, text='預測', command=self.Predict)
        self.predict_button.grid(row=0, column=1)

        self.clear_button = Button(self.root, text='清理', command=self.Clear)
        self.clear_button.grid(row=0, column=2)

        self.stamp_canvas = Canvas(self.root, width=150, height=150, bg='white', highlightthickness=0, relief='ridge')
        self.stamp_canvas.grid(row=1, column=1, padx=10, pady=10)

        self.feature_canvas = Canvas(self.root, width=150, height=150, bg='white', highlightthickness=0, relief='ridge')
        self.feature_canvas.grid(row=1, column=2, padx=10, pady=10)

        self.prediction_label = Label(self.root, text="", fg='blue')
        self.prediction_label.grid(row=2, column=1, columnspan=2)

        self.root.mainloop()

    def Predict(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.show_stamp_image(file_path)
            predicted_label, img_array = predict_stamp(file_path, model, label_encoder)
            self.prediction_label.config(text=f"This stamp belongs to: {predicted_label}")
            self.show_feature_map(img_array)

    def show_stamp_image(self, img_path):
        img = Image.open(img_path).resize((150, 150), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(img)
        self.stamp_canvas.create_image(0, 0, anchor=NW, image=img)
        self.stamp_canvas.image = img

    def show_feature_map(self, img_array):
        feature_index = 8
        layer_name = 'conv1_conv'
        img = get_feature_map_image(model, img_array, layer_name, feature_index)

        img = ImageTk.PhotoImage(img)
        self.feature_canvas.create_image(0, 0, anchor=NW, image=img)
        self.feature_canvas.image = img

    def Clear(self):
        self.stamp_canvas.delete("all")
        self.feature_canvas.delete("all")
        self.prediction_label.config(text="")


if __name__ == '__main__':
    StampClassifierApp()
