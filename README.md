# captcha-solver
 Different stages of a captcha solver, evolving over the semester.

Command to run the inference on the Rpi:

python3 classify_rpi.py --symbols s3.txt --length 6 --tflite_model m5.tflite --captcha_dir test_data_p2 --output_file output_tflite_rpi_5.txt
