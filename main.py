import os
import sys
import wget
import cv2
import numpy
import time
from matplotlib import pyplot

# Dimension of a Sudoku puzzle
matrix_size = 9

def main():
    if len(sys.argv) < 2:
        raise Exception("We need the name of the image file")
    input_image = sys.argv[1]
    image_path, image_file = os.path.split(input_image)
    image_file_name, image_file_extension = os.path.splitext(image_file)
    image = Image.open(input_image)
    image_mono = util.preprocess_image(image)
    image_mono.save(os.path.join(image_path, image_file_name + '_mono' + image_file_extension))
    horizontal_lines, vertical_lines = util.find_lines(image_mono)
    if len(horizontal_lines)-1 != matrix_size or len(vertical_lines)-1 != matrix_size:
        raise Exception('Could not find 10 vertical and 10 horizontal lines in the image')
    cell_image_boundaries = util.get_cell_image_boundaries(horizontal_lines, vertical_lines)
    util.trim_cell_images(image_mono, cell_image_boundaries)

if __name__ == '__main__':
    main()
