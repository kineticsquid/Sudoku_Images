from PIL import Image
from PIL import ImageDraw
import sys
import json
import util

def main():
    if len(sys.argv) < 3:
        raise Exception("Input parameter 1: Test image to use for processing, parameter 2: folder for image metrics json.")
    image_name = sys.argv[1]
    metrics_folder = sys.argv[2]
    x_metrics_file = open(metrics_folder + "x.json", "r")
    y_metrics_file = open(metrics_folder + "y.json", "r")
    x_metrics = json.loads(x_metrics_file.read())
    y_metrics = json.loads(y_metrics_file.read())
    image_mono = util.preprocess_image(image_name)
    image_mono.show()
    # util.print_image(image_mono, None)
    horizontal_lines, vertical_lines = util.find_lines(image_mono)
    show_lines_on_image(image_mono, horizontal_lines, vertical_lines)
    cell_image_boundaries = util.get_cell_image_boundaries(horizontal_lines, vertical_lines)
    show_cell_images(image_mono, cell_image_boundaries)
    rows, columns = cell_image_boundaries.shape
    util.trim_cell_images(image_mono, cell_image_boundaries)
    matrix = []
    for row in range(rows):
        new_row = []
        for column in range(columns):
            if sum(cell_image_boundaries[row, column]) > 0:
                number_image = image_mono.crop((cell_image_boundaries[row, column]))
                number_image.show()
                number = util.get_number_from_image(number_image, x_metrics, y_metrics)
                new_row.append(number)
            else:
                new_row.append(0)
        matrix.append(new_row)
    print(matrix)


"""
Utility for putting horizontal and vertical lines back on the source image, for debugging
"""
def show_lines_on_image(image_mono, horizontal_lines, vertical_lines):
    new_image = image_mono.copy()
    new_image_width, new_image_height = new_image.size
    d = ImageDraw.Draw(new_image)
    teal = (66, 245, 239)
    for h_line in horizontal_lines:
        d.rectangle((0, h_line[0], new_image_width, h_line[1]), outline="#888888", fill="#888888")
    for v_line in vertical_lines:
        d.rectangle((v_line[0], 0, v_line[1], new_image_height), outline="#888888", fill="#888888")
    new_image.show()

"""
Utility routine for debugging purposes only
"""
def show_cell_images(image_mono, cell_image_boundaries):
    rows, columns = cell_image_boundaries.shape
    for row in range(rows):
        for column in range(columns):
            if sum(cell_image_boundaries[row, column]) > 0:
                number_image = image_mono.crop((cell_image_boundaries[row, column]))
                number_image.show()

if __name__ == '__main__':
    main()
