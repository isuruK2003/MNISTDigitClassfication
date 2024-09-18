import pickle
import numpy as np
from PIL import Image, ImageOps
from trainer import predict


def get_model_data(filename):
  try:
    model_data = {}
    with open(filename, 'rb') as file:
      model_data = pickle.load(file)
  except FileNotFoundError:
      print("Pickled file not found! Please train the model and try again!")
  finally:
      return model_data


def get_image_data(filepath, auto_invert_color = False):
  image = Image.open(filepath)
  image = image.resize((28, 28))
  image = ImageOps.grayscale(image)

  image_data = np.array(image).T
  image_data = image_data.flatten('F')
  image_data = np.array([image_data])

  if auto_invert_color and np.mean(image_data) > 125:
    image_data = 255 - image_data

  return image_data


def predict_image(x_array, true_num):
  model = get_model_data(f"models/model_for_{true_num}")
  scaler = model["scaler"]
  w_array = model["w_array"]

  x_array = scaler.transform(x_array)
  prediction_val = predict(x_array, w_array)
  prediction = True if prediction_val > 0.5 else False
  return prediction, prediction_val


def classify(image_data):
  prediction_vals = {str(i) : predict_image(image_data, i)[-1][0] for i in range(10)}
  prediction_vals_keys = sorted(prediction_vals, key=lambda k : prediction_vals[k], reverse=True)
  prediction_vals = {k:prediction_vals[k] for k in prediction_vals_keys}
  return list(prediction_vals.keys())[0], prediction_vals


def print_ascii_image(image_data, width = 28):
  density = " `.-'_,^=:;><+!rc*/z?sLTv#%&$NWQ@"
  chars = []
  max_pixel_val = max(image_data)
  min_pixel_val = min(image_data)

  for i in range(len(image_data)):
    end = "\n" if (i + 1) % width == 0 else " "
    normalized_val = (image_data[i] - min_pixel_val) / (max_pixel_val - min_pixel_val)
    char_index = int(normalized_val * (len(density) - 1))
    char = density[char_index]
    chars.append(f"{char}{end}")
  print("".join(chars))


def main():
  filepath = input("Enter image filepath: ")

  image_data = get_image_data(filepath, True)
  prediction, prediction_vals = classify(image_data)
  strength = (prediction_vals[prediction] / sum(prediction_vals.values())) * 100

  print_ascii_image(image_data[0])
  print(f"Prediction: {prediction}")
  print(f"Strength: {strength:.2f}%")
  print("Prediction Values :")
  print("\n".join((f"{key} : {val}" for key, val in prediction_vals.items())))


if __name__ == "__main__":
   main()
