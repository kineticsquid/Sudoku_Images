import cv2
import numpy
from matplotlib import pyplot
import sys
import json
import util


"""
Main routine to process the image and extract the sudoku matrix from it
"""


def extract_matrix_from_image(image, x_metrics, y_metrics):

    x_coords, y_coords = util.get_cell_boundaries(image)

    # Now extract the images of the cells (between the lines), based on the coordinates we just
    # calculated. When we have these extracted images, sum the pixel counts along the horizontal
    # and vertical axes. This will be used to compare with training data to identify the digits
    # in the cells

    pyplot.rcParams['figure.dpi'] = 50
    bw_threshold = 160
    # Make the image monochrome. If the original pixel value > threshold, 1, otherwise 0.
    (thresh, monochrome_image) = cv2.threshold(image, bw_threshold, 1, cv2.THRESH_BINARY_INV)

    puzzle_matrix = []
    for y_coord in y_coords:
        new_matrix_row = []
        for x_coord in x_coords:
            raw_image = monochrome_image[y_coord[0]:y_coord[1], x_coord[0]:x_coord[1]]
            image_height, image_width = raw_image.shape
            image_sum = raw_image.sum()
            image_density = image_sum / (image_width * image_height)
            # If the image density (% of black pixels in the image) is less than a certain threshold
            # we assume the cell is empty and return 0. This is not a test for 0 % since there can be
            # noise in the image. If above the threshold, then determine the number from training data
            if image_density < 0.001:
                number = 0
            else:
                # show_image(raw_image,
                #            title="Y - %s:%s. X - %s:%s" % (y_coord[0], y_coord[1], x_coord[0], x_coord[1]))
                cell_image = util.trim(raw_image)
                if cell_image is None:
                    number = 0
                else:
                    number = get_number_from_image(cell_image, x_metrics, y_metrics)
            print("Number: %s" % number)
            new_matrix_row.append(number)
        puzzle_matrix.append(new_matrix_row)
        print("\n")
    return puzzle_matrix


"""
This routine finds the digit from a cell image based on the training data
"""


def get_number_from_image(cell_image, x_metrics, y_metrics):
    # This is summing columns vertically
    x_counts = cell_image.sum(axis=0)
    # This is summing rows horizontally
    y_counts = cell_image.sum(axis=1)
    x_metrics_size = len(x_metrics["1"])
    y_metrics_size = len(y_metrics["1"])
    #normalize the size of the image arrays to match the metrics data
    x_histogram = morph_array_to_size(x_counts, x_metrics_size)
    y_histogram = morph_array_to_size(y_counts, y_metrics_size)
    # Now get percentages
    x_sum = sum(x_histogram)
    y_sum = sum(y_histogram)
    for n in range(x_histogram.size):
        x_histogram[n] = x_histogram[n]/x_sum
    for n in range(y_histogram.size):
        y_histogram[n] = y_histogram[n]/y_sum
    # Now calculate difference between x and y pixel distribution for this image and metrics data
    # the 1000 is there for readability
    x_distance = numpy.zeros((len(x_metrics)+1), dtype=float)
    for n in range(0, len(x_metrics)):
        x_distance[n] = diff_between_arrays(x_histogram, x_metrics[str(n)]) * 1000
    y_distance = numpy.zeros((len(y_metrics)+1), dtype=float)
    for n in range(0, len(y_metrics)):
        y_distance[n] = diff_between_arrays(y_histogram, y_metrics[str(n)]) * 1000

    # x_distance is an array that has the least squares distance between the x-axis histogram
    # of this image and the x-axis histograms of the training data [1..9]. y_distance is the
    # same thing along the y-axis. The algorithm to determine the digit in the image goes like this:
    # We first match using the y-axis histogram. Experiments have shown this is accurate for
    # everything except some confusion when a '1' is incorrectly recognized as a '2'. So,
    # when we see a '2', perform a second check using the x-axis to see which distance is less,
    # '1' or '2'.
    #
    # Other algorithms attempted:
    # - x_distance
    # - y_distance
    # - x_distance * y_distance
    current_y_distance_min = y_distance[0]
    number = 0
    # print("x histogram: %s" % x_histogram)
    # print("y histogram: %s" % y_histogram)
    for n in range(0, len(x_metrics)):
        print('n: %s\tx_distance: %.2f\ty_distance: %.2f' %
              (n, x_distance[n], y_distance[n]))
        if y_distance[n] < current_y_distance_min:
            current_y_distance_min = y_distance[n]
            number = n
    if number == 2:
        if x_distance[1] < x_distance[2]:
            number = 1

    return number

"""
The sizes of the images for the matrix cells vary, meaning the length of the image histogram arrays of the
pixel distributions vary in length. This routine changes the length of the image histogram array to 
match the length of the training data array. It also strips leading and trailing 0s from the input 
array, effectively trimming blank space from around the digit (to match what's in the training data).

Because detecting lines is imperfect, the images can have some noise around the edges, parts of the lines
we weren't able to remove, the trimming starts in the middle and works its way out. Note, this assumes 
that the number in the cell is roughly centered (pretty safe for computer generate Sudoku puzzles).
"""

def morph_array_to_size(array, to_size):
    from_array = array
    from_size = len(from_array)
    to_array = numpy.zeros((to_size), dtype=float)

    if from_size >= to_size:
        # for f in range(from_size):
        #     to_array[f] = from_array[f]
        for f in range(from_size):
            starting_to_element = int(to_size / from_size * f)
            ending_to_element = int(to_size / from_size * (f + 1))
            if starting_to_element == ending_to_element or \
                    ending_to_element - to_size / from_size * (f + 1) == 0:
                to_array[starting_to_element] += from_array[f]
            else:
                amt1_pct = from_size / to_size * f - int(from_size / to_size * f)
                amt2_pct = 1 - amt1_pct
                to_array[starting_to_element] += from_array[f] * amt1_pct
                to_array[ending_to_element] += from_array[f] * amt2_pct
    else:
        for t in range(to_size):
            starting_from_element = int(from_size / to_size * t)
            ending_from_element = int(from_size / to_size * (t + 1))
            if starting_from_element == ending_from_element or \
                    ending_from_element - from_size / to_size * (t + 1) == 0:
                to_array[t] += from_array[starting_from_element] * from_size / to_size
            else:
                amt1_pct = int(from_size / to_size * t) + 1 - from_size / to_size * t
                amt2_pct = from_size / to_size * (t + 1) - int(from_size / to_size * (t + 1))
                to_array[t] += from_array[starting_from_element] * amt1_pct
                to_array[t] += from_array[ending_from_element] * amt2_pct

    return to_array

def diff_between_arrays(a1, a2):
    diff = 0
    for n in range(len(a1)):
        diff += (a1[n] - a2[n]) * (a1[n] - a2[n])
    return diff

def main():
    if len(sys.argv) < 3:
        raise Exception("Inputs:\n\tparameter 1: Test image to use for processing.\n\tparameter 2: folder for image metrics json.")
    input_image = sys.argv[1]
    print("Image: %s" % input_image)
    metrics_folder = sys.argv[2]
    x_metrics_file = open(metrics_folder + "x.json", "r")
    y_metrics_file = open(metrics_folder + "y.json", "r")
    x_metrics = json.loads(x_metrics_file.read())
    y_metrics = json.loads(y_metrics_file.read())
    image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    matrix = extract_matrix_from_image(image, x_metrics, y_metrics)
    print(matrix)

if __name__ == '__main__':
    main()