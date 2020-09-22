from PIL import Image
import sys
import os
from os import listdir
import util
import cv2
import numpy
from matplotlib import pyplot

def main():
    if len(sys.argv) < 3:
        raise Exception("Input parameter 1: folder of input images.Input parameter 2: folder for output")
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    for image_file in listdir(input_folder):
        print("Processing: %s..." % image_file)
        image_file_name, image_file_extension = os.path.splitext(image_file)
        if image_file_extension == '' or image_file_name[0] == '.':
            print("Skipping, not an image file.")
        else:
            image = cv2.imread("%s/%s" % (input_folder, image_file), cv2.IMREAD_GRAYSCALE)
            x_coords, y_coords = util.get_cell_boundaries(image)
            bw_threshold = 160
            row = 1
            for y_coord in y_coords:
                column = 1
                for x_coord in x_coords:
                    cell_image = image[y_coord[0]:y_coord[1], x_coord[0]:x_coord[1]]
                    # util.show_image(cell_image)
                    # Make the image monochrome. If the original pixel value > threshold, 1, otherwise 0.
                    # We use 1 and 0 since this will make it easy to sum the pixels in each direction
                    (thresh, monochrome_image) = cv2.threshold(cell_image, bw_threshold,
                                                               1, cv2.THRESH_BINARY_INV)
                    util.show_image(monochrome_image)
                    # Save the image of this cell
                    new_file_name = "%s/%s/%s%s%s%s" % (output_folder, column,
                                                        image_file_name, row, column, image_file_extension)
                    write_succeeded = cv2.imwrite(new_file_name, monochrome_image)


                    column += 1

                row += 1

        print("Finished: %s" % image_file)

if __name__ == '__main__':
    main()
