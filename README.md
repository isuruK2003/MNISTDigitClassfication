# MNIST Digit Classification

This project demonstrates image classification from scratch using the MNIST dataset. It includes both binary and multinomial classifiers implemented in Python.

## No libraries, all from scratch

In this project, image classification models are implemented from scratch without using high-level libraries like TensorFlow or PyTorch. The focus is on understanding the fundamental concepts of machine learning and image processing. The project includes:

1. **Binary Classifier**: A logistic regression model to classify images as either digit '0' or '1'.
2. **Multinomial Classifier**: An extension of the binary classifier to handle all ten digits (0-9) using a multinomial logistic regression approach.

## Binary Classifier

### Training

The binary classifier is trained using logistic regression. The training process involves:

1. Loading the dataset from `mnist_train_small.csv`.
2. Preprocessing the data by scaling and splitting it into training and testing sets.
3. Training the model using gradient descent to minimize the cost function.

The training code is implemented in `trainer.py`.

### Prediction

The prediction process involves:

1. Loading the trained model from the `models` directory.
2. Preprocessing the input image.
3. Predicting the class of the input image using the trained model.

The prediction code is implemented in `predictor.py`.

## Multinomial Classifier

The multinomial classifier is implemented in the Jupyter notebook `MNIST_Digit_Classification_Multinomial_Improved.ipynb`. It uses a similar approach to the binary classifier but extends it to handle multiple classes.

## Running the Code

To train the binary classifier, run:

```sh
python binary-classifier/trainer.py
```

To predict using the binary classifier, run:

```sh
python binary-classifier/predictor.py
```

For the multinomial classifier, open the Jupyter notebook and execute the cells.

## Dependencies

- Python 3.x
- NumPy
- Pandas
- scikit-learn
- Pillow
- Matplotlib

Install the dependencies using:

```sh
pip install numpy pandas scikit-learn pillow matplotlib
```