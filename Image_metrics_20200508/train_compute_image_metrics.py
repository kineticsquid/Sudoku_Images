from PIL import Image
import sys
import os
from os import listdir
import numpy
import util
import json
import io

black = 0
white = 255
metrics_array_size = 20
matrix_size = 9

def output_json(matrix, output_file_name):
    output = {}
    numbers, count = matrix.shape
    for number in range(numbers):
        if number > 0:
            output[number] = matrix[number].tolist()
    print(json.dumps(output, indent=4))
    output_file = open(output_file_name, 'w')
    output_file.write(json.dumps(output, indent=4))
    output_file.close()



def main():
    if len(sys.argv) < 3:
        raise Exception("Input parameter 1: folder of folders of training data images. Input parameter 2: folder for JSON output representing image metrics.")
    # input fold of images organized into folders named by the number in in the image
    input_folder = sys.argv[1]
    # output folder for json metrics
    output_folder = sys.argv[2]
    # matrix to keep totals for where bits are found in the example images on both x and y axes
    x_metrics = numpy.zeros((matrix_size+1, metrics_array_size), dtype=float)
    y_metrics = numpy.zeros((matrix_size+1, metrics_array_size), dtype=float)

    for image_folder_name in listdir(input_folder):
        image_folder = input_folder + image_folder_name
        if os.path.isdir(image_folder):
            print("Processing: %s..." % image_folder_name)
            # current_number is the number in the images in this folder
            current_number = int(image_folder_name)
            for image_file in listdir(image_folder):
                image_file_name, image_file_extension = os.path.splitext(image_file)
                if image_file_extension == '' or image_file_name[0] == '.':
                    print("Skipping, not an image file.")
                else:
                    image = Image.open("%s/%s" % (image_folder, image_file))
                    # arrays to hold the x and y axis totals for this image
                    x_histogram, y_histogram = util.count_pixels(image)
                    # because images come in different sizes we need to normalize the length of the array
                    # of counts to be a constant
                    normalized_x = util.morph_array_to_size(x_histogram, metrics_array_size)
                    normalized_y = util.morph_array_to_size(y_histogram, metrics_array_size)
                    print("%s:\tx: %s, %s\ty: %s, %s" % (image_file, sum(x_histogram), sum(normalized_x),
                                                         sum(y_histogram), sum(normalized_y)))
                    print("x:\t%s" % normalized_x)
                    print("y:\t%s" % normalized_y)

                    # now add totals from this image to totals
                    for n in range(metrics_array_size):
                        x_metrics[current_number][n] += normalized_x[n]
                        y_metrics[current_number][n] += normalized_y[n]
                    print("x %s total:\t%s" % (current_number, x_metrics[current_number]))
                    print("y %s total:\t%s" % (current_number, y_metrics[current_number]))
                    print("Finished image %s" % image_file)

            print("Finished: %s" % image_folder)
            # Now that we have counts for all images of this number, we need to compute the % distribution
            print("x:")
            print(x_metrics[current_number])
            print("y:")
            print(y_metrics[current_number])
            total = sum(x_metrics[current_number])
            for n in range(x_metrics[current_number].size):
                x_metrics[current_number][n] = x_metrics[current_number][n] / total
            total = sum(y_metrics[current_number])
            for n in range(y_metrics[current_number].size):
                y_metrics[current_number][n] = y_metrics[current_number][n] / total
            print("x:")
            print(x_metrics[current_number])
            print("y:")
            print(y_metrics[current_number])

    output_json(x_metrics, output_folder + "x.json")
    output_json(y_metrics, output_folder + "y.json")

if __name__ == '__main__':
    main()
