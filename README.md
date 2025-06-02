# IMAGE-CLASSIFICATION-MODEL

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: RITZ LONGJAM

*INTERN ID*: CT08DA250

*DOMAIN*: MACHINE LEARNING

*DURATION*: 8 WEEKS

*MENTOR*: NEELA SANTHOSH KUMAR

#TASK DESCRIPTION: Building a Convolutional Neural Network for Image Classification

In this task, we developed a Convolutional Neural Network (CNN) model for image classification using the TensorFlow framework with Python as the programming language. The entire workflow was implemented and executed on Google Colab, a cloud-based Jupyter notebook environment that supports free GPU acceleration, making it suitable for deep learning tasks. The objective was to build a functional CNN model capable of classifying images from the CIFAR-10 dataset, and then evaluate its performance on unseen test data.

Dataset and Preprocessing

The CIFAR-10 dataset is a widely used benchmark in computer vision, containing 60,000 color images of size 32x32 pixels spread across 10 different classes such as airplane, automobile, bird, cat, and more. The dataset is split into 50,000 training images and 10,000 test images. For this project, the images were loaded directly from TensorFlow's Keras datasets module.

Before training, we normalized the pixel values to a range between 0 and 1 to speed up the training process and improve model performance. The label arrays were reshaped into a 1D format to suit the input format expected by TensorFlow’s loss function. A helper function was also defined to display sample images from the dataset along with their class labels.

Model Architecture

The CNN model was created using Keras’ Sequential API. The architecture began with two convolutional layers (with 32 and 64 filters) using ReLU activation, each followed by a max-pooling layer. These layers help extract spatial features and reduce dimensionality. The output of the convolutional layers was then flattened and passed through a dense layer with 64 units and ReLU activation. Finally, a softmax output layer with 10 neurons was added to predict class probabilities.

Model Compilation and Training

We compiled the model using the Adam optimizer and sparse categorical cross-entropy as the loss function—suitable for multi-class classification. The model was trained for 10 epochs using the training data, and its performance was validated on the test set during each epoch. We stored the training history to later visualize accuracy trends.
Evaluation and Visualization

After training, the model was evaluated on the test set. The test accuracy was printed to show the model’s performance on unseen data. Predictions were generated for the test set, and we calculated a classification report that included precision, recall, and F1-score for each class. A confusion matrix was also plotted using Seaborn to visualize which classes were most often confused by the model.

To make the results more interpretable, we created a function to display test images with their actual and predicted labels. Finally, we plotted the training and validation accuracy curves to observe the model’s learning pattern over epochs.

Tools and Platform Used

•	Programming Language: Python

•	Libraries: TensorFlow, Keras, NumPy, Matplotlib, Seaborn, scikit-learn

•	Editor/Platform: Google Colab with GPU support

•	Dataset: CIFAR-10 (via TensorFlow)

Applications

This task highlights how CNNs can be used for image classification, which has broad applications in real-world scenarios. These include facial recognition, traffic sign detection, medical imaging analysis, quality control in manufacturing, and many others. This task provides a foundation that can be expanded with more complex networks or transfer learning for advanced use cases.

*OUTPUT*

<img width="374" alt="Image" src="https://github.com/user-attachments/assets/7c7032b6-1336-45a9-be91-0b6557a1e2f3" />

<img width="227" alt="Image" src="https://github.com/user-attachments/assets/810ff179-d50e-4052-b487-735771d02942" />

<img width="127" alt="Image" src="https://github.com/user-attachments/assets/39d7f80f-75a3-48a5-abd4-a7ec4f74cf3f" />

<img width="292" alt="Image" src="https://github.com/user-attachments/assets/f46e9bee-2b37-4d26-9c62-8df96764a4cc" />
