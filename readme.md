### Main Notebooks
- https://dataplatform.cloud.ibm.com/projects/938025da-4451-4b2c-85c9-a5ed6d65ca7d/assets?context=cpdaas
### Resources
- https://pillow.readthedocs.io/en/stable/
- https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
- https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
- https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html
- https://nanonets.com/blog/ocr-with-tesseract/

### Training
These two routines are used for generating the training data used to recignize numbers extracted from Sudoku puzzle matrices.

`train_extract_images.py` processes images in an input directory and extracts digits from cells as individual trimmed images and puts these images files in the output file organized in directories by the digit

`train_compute_image_metrics.py` uses these individual digit images to compute the training data, which goes in the output folder.

Note, these routines need to be re-written to make better use of cv2 and numpy.