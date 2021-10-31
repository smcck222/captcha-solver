import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import tflite_runtime.interpreter as tflite

import os
import cv2
import numpy
import string
import random
import argparse

# args : symbols, length, tflite_model, captcha_dir, output_file
# RUN inside dir. 
# python3 classify_rpi.py --symbols s3.txt --length 6 --tflite_model m5.tflite --captcha_dir test_data_p2 --output_file output_tflite_rpi_5.txt

def get_symbol(index):
  return(captcha_symbols[index])

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--symbols', help='Symbols set', type=str)
  parser.add_argument('--length', help='Length of captchas', type=int)
  parser.add_argument('--tflite_model', help='trained model', type=str)
  parser.add_argument('--captcha_dir', help='test set', type=str)
  parser.add_argument('--output_file', help='where to write the output', type=str)
  args = parser.parse_args()
  
  if args.symbols is None:
    print("Please specify the symbol set")
    exit(1)
  
  if args.length is None:
    print("Please specify the captcha length")
    exit(1)
  
  if args.tflite_model is None:
    print("Please specify the tflite model")
    exit(1)

  if args.captcha_dir is None:
        print("Please specify the directory for test images")
        exit(1)

  if args.output_file is None:
        print("Please specify the output file")
        exit(1)
  
  symbols_file = open(args.symbols, 'r')
  captcha_symbols = symbols_file.readline().strip()
  symbols_file.close()
  captcha_symbols = captcha_symbols + '@'
  
  output_dict = {}
  # Output is written into dict with format {filename.png : captcha}
  
  print("Classifying captchas with symbol set {" + captcha_symbols + "}")
  
  # Load the TFLite model and allocate tensors.
  # Load the TFLite model and allocate tensors.
  interpreter = tflite.Interpreter(model_path = args.tflite_model)
  
  # Get input and output tensors.
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  
  for x in os.listdir(args.captcha_dir):
    raw_data = cv2.imread(os.path.join(args.captcha_dir, x))
    rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
    image = numpy.array(rgb_data) / 255.0
    (c, h, w) = image.shape
    image = image.reshape([-1, c, h, w])
    
    input_data = numpy.array(image, dtype=numpy.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    pred = ""
    for i in range(args.length):
      output_data = interpreter.get_tensor(output_details[i]['index'])
      results = np.squeeze(output_data)
      index = np.argmax(results)
      captcha_at_index = get_symbol(index)
      pred = pred + captcha_at_index
    
    truncated_pred = pred.replace('/', ':')
    truncated_pred = truncated_pred.replace('@', '')

    output_dict[x] = truncated_pred
    print('Classified ' + x + 'as: ' + output_dict[x])

  with open(args.output, 'w') as output_file:
    output_file.write("scherian"+"\n")
    for key in output_dict:
      output_file.write(key + "," + output_dict[key] + "\n")

if __name__ == '__main__':
    main()

