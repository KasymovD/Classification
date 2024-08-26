# Classification
Fake Stamp Classification Project
Project Description
This project aims to classify fake stamp images using a deep neural network based on the ResNet50 architecture. The model is trained on data representing different types of fake stamps and is able to detect the class of a stamp given an input image.

Features
Model Architecture: ResNet50 pre-trained on ImageNet.
Optimization: Using Adam with an initial learning rate of 0.0001.
Monitoring: Early stopping with patience=10, configured to monitor the validation loss (val_loss), with the best weights restored.
Extra Layers: Added Dense layers with Dropout to improve the generalization ability of the model.
Saving Classes: Saved class labels to classes.pkl using joblib for later use in predictions.
Usage
Training the model:

The model is trained on the data in the Processed_Dataset/公司 and Processed_Dataset/關防-整理好的 directories.
Training will automatically stop if the model does not improve over 10 epochs.
Loading and using the model:

After training, the model is saved to the stamp_classification_model_resnet50.h5 file.
Class labels are saved to the classes.pkl file.
Run
To run training, run:

bash
Copy code
python model.py
Dependencies
TensorFlow
Keras
NumPy
scikit-learn
Pillow
joblib