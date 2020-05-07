# Intuit-TeamLoki
OCR and Computer Vision 
Implementation of EAST with Tesseract

To run our implementation:
  1. Clone EAST folder 
  2. Get checkpoint east_icdar2015_resnet_v1_50_rbox/ from https://github.com/argman/EAST/blob/master/ and move it to the EAST folder
  3. Download our realistic noisy W-2 images from: 
  3. In the same directory as EAST create folder images/ and place the W-2 images in this file

To run python file, run this command from terminal in the EAST folder:

python eval.py --test_data_path=images/ --gpu_list=0 --checkpoint_path=east_icdar2015_resnet_v1_50_rbox/ \
--output_dir=tmp/

To run on Jupyter notebook:
1. Open eval.ipynb 
2. run all cells
