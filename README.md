# Intuit-TeamLoki
OCR and Computer Vision 
Implementation of EAST with Tesseract

To run our implementation:
  1. Clone EAST folder 
  2. Get checkpoint east_icdar2015_resnet_v1_50_rbox/ from https://github.com/argman/EAST/ and move it to the EAST folder
  3. Download our realistic noisy W-2 images from: https://drive.google.com/drive/folders/1GW76o8HqllxT4UInX5yXWP7V7LAm-M8q?usp=sharing
  4. In the same directory as EAST create folder images/ and place the W-2 images in this file

You can now either run the code as a python file or Jupyter notebook.

To run python file, run this command from terminal in the EAST folder:

python eval.py --test_data_path=images/ --gpu_list=0 --checkpoint_path=east_icdar2015_resnet_v1_50_rbox/ \
--output_dir=tmp/

To run on Jupyter notebook:

1. Open eval.ipynb from the EAST folder
2. Change the paths in the second cell to match your paths
3. Run all cells


Troubleshoot:
If EAST doesn't seem to be running correctly, directly clone the EAST folder from https://github.com/argman/EAST/
