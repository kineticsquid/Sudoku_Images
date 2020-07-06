from PIL import Image
import sys
import os
from os import listdir
import util

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
            image = Image.open(input_folder + image_file)
            image_mono = util.preprocess_image(image)
            image_mono.show()
            # util.print_image(image_mono, None)
            horizontal_lines, vertical_lines = util.find_lines(image_mono)
            cell_image_boundaries = util.get_cell_image_boundaries(horizontal_lines, vertical_lines)
            rows, columns = cell_image_boundaries.shape
            util.trim_cell_images(image_mono, cell_image_boundaries)
            for row in range(rows):
                for column in range(columns):
                    if sum(cell_image_boundaries[row, column]) > 0:
                        new_file_name = "%s%s%s%s%s" % (output_folder, image_file_name, row, column, image_file_extension)
                        image_mono.crop((cell_image_boundaries[row, column])).save(new_file_name)
                        print("Saved %s" % new_file_name)
        print("Finished: %s" % image_file)

if __name__ == '__main__':
    main()
