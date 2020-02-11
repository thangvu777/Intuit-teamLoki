#!/bin/bash
# My first script
# A script to automate tesseract
# Doesn't work with raw pdf's 

echo "#########  TESSERACT SCRIPT  ########"
echo
echo "Enter input directory: "
read inD
cd $inD
echo "Changed into /$inD"
for file in $inD/*
  do
  	find . -name '*.jpg' -exec tesseract {} {} \; 
  	#tesseract "$file" "$file"
  done
echo 
echo "#########  END OF TESSERACT SCRIPT  ########"