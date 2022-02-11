import cv2
import numpy
from matplotlib import pyplot


"""
Main routine to process the image and extract the sudoku matrix from it
"""


def extract_matrix_from_image(image, training_metrics):

    x_coords, y_coords = get_cell_boundaries(image)

    # Now extract the images of the cells (between the lines), based on the coordinates we just
    # calculated. When we have these extracted images, sum the pixel counts along the horizontal
    # and vertical axes. This will be used to compare with training data to identify the digits
    # in the cells

    pyplot.rcParams['figure.dpi'] = 50
    bw_threshold = 160
    # Make the image monochrome. If the original pixel value > threshold, 1, otherwise 0.
    (thresh, monochrome_image) = cv2.threshold(image, bw_threshold, 1, cv2.THRESH_BINARY_INV)

    puzzle_matrix = []
    row = 0
    for y_coord in y_coords:
        column = 0
        new_matrix_row = []
        for x_coord in x_coords:
            raw_image = monochrome_image[y_coord[0]:y_coord[1], x_coord[0]:x_coord[1]]
            image_height, image_width = raw_image.shape
            image_sum = raw_image.sum()
            image_density = image_sum / (image_width * image_height)
            # If the image density (% of black pixels in the image) is less than a certain threshold
            # we assume the cell is empty and return 0. This is not a test for 0 % since there can be
            # noise in the image. Or if the density is 1 it means it's completely black, so mark it
            # as zero also. Doing this here as it then makes the trimming easier.
            if image_density < 0.001 or image_density == 1:
                number = 0
            else:
                # show_image(raw_image,
                #            title="Y - %s:%s. X - %s:%s" % (y_coord[0], y_coord[1], x_coord[0], x_coord[1]))
                cell_image = trim(raw_image)
                if cell_image is None:
                    number = 0
                else:
                    number = get_number_from_image(cell_image, training_metrics)
            print("Row: %s\tColumn: %s\tValue: %s\n" % (row, column, number))
            new_matrix_row.append(number)
            column += 1
        puzzle_matrix.append(new_matrix_row)
        row += 1

    # now check for extra blank rows, likely due to not correctly identifying the lines
    # that define the matrix
    trim_matrix(puzzle_matrix)

    return puzzle_matrix

"""
This routine checks to see if the size of the matrix is > 9x9. If so, it looks for rows or columns
on the edge of the materix that are all 0s (blanks). Most likely this is due to noise at the edge of the
image that is creating additional matrix lines leading to additional matrix cells. The content of these
cells should be blank (0), so this routine removes them.
"""
def trim_matrix(puzzle_matrix):

    def sum_column(matrix, index):
        sum = 0
        for row in matrix:
            sum += row[index]
        return sum

    def remove_column_from_row(matrix, index):
        new_matrix = []
        for row in matrix:
            row.pop(index)

    done = False
    while not done and (len(puzzle_matrix) > 9 or len(puzzle_matrix[0]) > 9):
        done = True
        if sum(puzzle_matrix[0]) == 0:
            puzzle_matrix.pop(0)
            done = False
        if sum(puzzle_matrix[len(puzzle_matrix)-1]) == 0:
            puzzle_matrix.pop(len(puzzle_matrix)-1)
            done = False
        if sum_column(puzzle_matrix,0) == 0:
            remove_column_from_row(puzzle_matrix,0)
            done = False
        if sum_column(puzzle_matrix, len(puzzle_matrix[0])-1) == 0:
            remove_column_from_row(puzzle_matrix, len(puzzle_matrix[0])-1)
            done = False


"""
This routine finds the digit from a cell image based on the training data. For a description of the format
of the training date, see commentary in 'train_comput_image_metrics.py'.
"""


def get_number_from_image(cell_image, training_metrics):
    # todo: start here testing this with the training data image
    width, height = cell_image.shape
    # This calculates the % of cells that are part of the digit at each point on the x axis
    image_x_dist = cell_image.sum(axis=0)/(width*height) * 100
    # This does the same along the y axis.
    image_y_dist = cell_image.sum(axis=1)/(width*height) * 100
    #normalize the size of the image arrays to match the metrics data
    image_x_dist_normalized = morph_array_to_size(image_x_dist,
                                                       training_metrics['normalized_size']['x'])
    image_y_dist_normalized = morph_array_to_size(image_y_dist,
                                                       training_metrics['normalized_size']['y'])
    print("x_dist")
    print(image_x_dist.tolist())
    print("y_dist")
    print(image_y_dist.tolist())
    print("x_dist_normalized")
    print(image_x_dist_normalized)
    print("y_dist_normalized")
    print(image_y_dist_normalized)
    print("Training image, x-distance, y-distance, product")

    nearest_number = None
    nearest_x = None
    nearest_y = None
    for n in range(1, 10):
        for entry in training_metrics[str(n)]['normalized_data']:
            x_distance = diff_between_arrays(image_x_dist_normalized, entry['x'])
            y_distance = diff_between_arrays(image_y_dist_normalized, entry['y'])
            print("%s, %.3f, %.3f, %.3f" %
                  (entry['file_name'], x_distance, y_distance, x_distance * y_distance))

            # To compute the closest match. First use the product of least squares calculations
            # of x and y axis pixel distributions. If the best fit is 8 (which often matches other
            # numbers, instead check for best match only along the y axis.
            if nearest_number is None:
                nearest_number = n
                nearest_distance = x_distance * y_distance
                nearest_x = n
                nearest_x_distance = x_distance
                nearest_y = n
                nearest_y_distance = y_distance
            else:
                if x_distance * y_distance < nearest_distance:
                    nearest_number = n
                    nearest_distance = x_distance * y_distance
                if x_distance < nearest_x_distance:
                    nearest_x = n
                    nearest_x_distance = x_distance
                if y_distance < nearest_y_distance:
                    nearest_y = n
                    nearest_y_distance = y_distance

    if nearest_number == 8:
        nearest_number = nearest_y

    return nearest_number

"""
Routine to calculate the difference between two arrays
"""

def diff_between_arrays(a1, a2):
    diff = 0
    for n in range(len(a1)):
        diff += numpy.sqrt((a1[n] - a2[n]) * (a1[n] - a2[n]))
    return diff

"""
Routine to redistribute values in an array to an array of different length. This version is more
understandable than the original.

Purpose is to take an array of values of one length and redistribute those values over an
array of a different length. The approach is follows.
1. Create an intermediate array of length that both divide evenly into it. If the length of the 'from' 
array is 'f' and then length of the 'to' array is 't', create the intermediate array of length f*t.
2. Distribute the values of the 'from' array over the intermediate array 
  - Add t copies of the value of each of the elements in the 'from' array / t
3. Calculate the 'to' array by adding the values of each group of f values. 
"""
def morph_array_to_size(from_array, to_size):
    from_size = len(from_array)
    to_array = numpy.zeros((to_size), dtype=float)
    intermediate_array = numpy.zeros((from_size * to_size), dtype=float)

    intermediate_count = 0
    for element in from_array:
        for t in range(0, to_size):
            intermediate_array[intermediate_count] = element/to_size
            intermediate_count += 1

    for t in range(0, to_size):
        to_array[t] = numpy.sum(intermediate_array[t*from_size:(t+1)*from_size])
    return_value = to_array.tolist()

    return return_value

"""
Routine to redistribute values in an array to an array of different length. This is the original 
version
"""
def morph_array_to_size2(from_array, to_size):
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

"""
Routine to display an image and optionally a title
"""


def show_image(image, title=None, color=False):
    if color is True:
        pyplot.imshow(image)
    else:
        pyplot.imshow(image, cmap='Greys_r')
    if title is not None:
        pyplot.title(title)
        pyplot.show()


"""
Routine to add lines to an image. The source image must be color for the lines to show up.
"""


def show_lines(image, lines, title=None):
    # We get image height and width this way since a black and white image returns (height, width) and a color image returns (heigh, width, depth)
    image_shape = image.shape
    image_height = image_shape[0]
    image_width = image_shape[1]

    image_color_copy = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)

    color = (0, 0, 192)
    thickness = 5
    for line in lines:
        try:
            start_point = (line[0][0], line[0][1])
            end_point = (line[0][2], line[0][3])
            cv2.line(image_color_copy, start_point, end_point, color, thickness)
        except Exception as e:
            print(line)
            print(e)
    show_image(image_color_copy, title=title, color=True)


"""
Routine to find lines in an image. It uses HoughlinesP to find the lines. Because the alg
finds horizontal lines first and 'occupies' the pixels of these lines. Vertical lines then have gaps   
where the horizontal lines cross. The gaps inhibit the finding of the vertical lines. So, process is
to scan the image. Extract the horizontal lines. Then rotate the image 90 degrees and extract those
horizontal lines (which are the actual vertical ones).
"""


def find_lines(image):

    image_height, image_width = image.shape
    minimum_side = min(image_height, image_width)
    min_line_length = int(minimum_side / 2)
    max_line_gap = int(minimum_side / 100)
    threshold = minimum_side * 2

    done = False
    while not done:
        lines = cv2.HoughLinesP(image, 1, numpy.pi / 180, threshold=threshold, minLineLength=min_line_length,
                                maxLineGap=max_line_gap)
        horizontal_lines, vertical_lines, rejected_lines = separate_lines(lines)
        print("\nThreshold: %s" % threshold)
        print("Horizontal lines: %s" % len(horizontal_lines))
        print("Vertical lines: %s" % len(vertical_lines))
        print("Rejected lines: %s" % len(rejected_lines))
        if enough_lines(horizontal_lines, vertical_lines):
            done = True
        elif threshold > 10:
            threshold = int(threshold / 2)
        else:
            done = True

    return horizontal_lines, vertical_lines

"""
Routine to separate out and return only the horizontal and vertical lines and a separate list of those
lines rejected.
"""

def separate_lines(lines):
    margin_of_error = numpy.pi / 45
    horizontal_lines = []
    vertical_lines = []
    rejected_lines = []
    if lines is not None:
        for line in lines:
            x1 = line[0][0]
            y1 = line[0][1]
            x2 = line[0][2]
            y2 = line[0][3]
            sin_theta = (x1 - x2) / numpy.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
            theta = numpy.arcsin(sin_theta)
            if abs(theta) <= margin_of_error or abs(abs(theta) - numpy.pi) <= margin_of_error:
                # vertical line
                vertical_lines.append(line)
            elif abs(abs(theta) - numpy.pi / 2) <= margin_of_error or abs(
                    abs(theta) - numpy.pi * 3 / 2) <= margin_of_error:
                # horizontal line
                horizontal_lines.append(line)
            else:
                rejected_lines.append(line)
    return horizontal_lines, vertical_lines, rejected_lines

"""
Routine to invert image (black and white)
"""


def invert_image(image):
    bw_threshold = 160
    (thresh, inverted_image) = cv2.threshold(image, bw_threshold, 255, cv2.THRESH_BINARY_INV)
    return inverted_image


"""
Routine to pre-process an image before attempting to identify lines and extract digits. Among other things,
make it monochromatic.

We'll use this once we deal with real photos vs computer generated images/graphics.
"""


def preprocess_image(image):

    return image

"""
Routine to determine if we've found enough lines 
"""
def enough_lines(horizontal_lines, vertical_lines):
    if len(horizontal_lines) > 0:
        if len(vertical_lines) > 0:
            if len(horizontal_lines) > len(vertical_lines):
                if len(vertical_lines) / len(horizontal_lines) > 0.5:
                    done = True
                else:
                    done = False
            elif len(vertical_lines) > len(horizontal_lines):
                if len(horizontal_lines) / len(vertical_lines) > 0.5:
                    done = True
                else:
                    done = False
            else:
                done = True
        else:
            done = False
    else:
        done = False
    return done
"""
This routine first trims traces of lines left at the edges of the image and then trims the blank
space to leave just the digit. Because there is noise in the images, we use the x_min and y_min
values to represent blank space. If the trimming results in nothing left, the routine returns None,
which gets interpreted as a blank cell.
"""

def trim(raw_image):
    # This is summing columns vertically
    x_counts = raw_image.sum(axis=0)
    # This is summing rows horizontally
    y_counts = raw_image.sum(axis=1)
    x_min = min(x_counts)
    y_min = min(y_counts)
    x_start = 0
    y_start = 0
    x_end = len(x_counts) - 1
    y_end = len(y_counts) - 1

    # Remove residual matrix line noise from around the edges of the image
    done = False
    while not done:
        # trim residual line noise from the left
        while x_counts[x_start] > x_min:
            x_start += 1
        #trim residual line noise from the top
        while y_counts[y_start] > y_min:
            y_start += 1
        #trim residual line noise from the right
        while x_counts[x_end] > x_min:
            x_end -= 1
        #trim residual line noise from the bottom
        while y_counts[y_end] > y_min:
            y_end -= 1
        trimmed_image = raw_image[y_start:y_end+1, x_start:x_end+1]
        x_counts = trimmed_image.sum(axis=0)
        y_counts = trimmed_image.sum(axis=1)
        x_min = min(x_counts)
        y_min = min(y_counts)
        x_start = 0
        y_start = 0
        x_end = len(x_counts) - 1
        y_end = len(y_counts) - 1

        if x_counts[x_start] == 0 and x_counts[x_end] == 0 \
                and y_counts[y_start] == 0 and y_counts[y_end] == 0:
            done = True

    # Now what we should have is an image with a digit in the middle surrounded by
    # some indeterminate amount of white space. Need to remove that white space.
    #
    # Remove blank space on the left
    while x_counts[x_start] == x_min and x_start != x_end:
        x_start += 1
    # If x_start == x_end after removing the line noise from the edges, it means the cell
    # is empty, so return None.
    if x_start == x_end:
        trimmed_image = None
    # Otherwise, finish removing the blank space around the digit.
    else:
        # Remove the blank space on the right
        while x_counts[x_end] == x_min and x_end != x_start:
            x_end -= 1
        # Now remove blank space on the top
        while y_counts[y_start] == y_min and y_end != y_start:
            y_start += 1
        # If y_start == y_end, it means the cell is empty, so return None.
        if y_start == y_end:
            trimmed_image = None
        else:
            # Now remove blank space on the bottom
            while y_counts[y_end] == y_min:
                y_end -= 1
            trimmed_image = trimmed_image[y_start:y_end+1, x_start:x_end+1]

    return trimmed_image

"""
Routine to process an image to find the lines and then the inside boundaries of the cells that contain
the digits
"""

def get_cell_boundaries(image):
    # First, invert the image so that the lines and digits are white and the background black.
    # We need this for the cv.houghline algorithm to work.
    # show_image(image)
    inverted_image = invert_image(image)
    # show_image(inverted_image)
    image_height, image_width = inverted_image.shape
    print("Inverted image width:\t%s" % image_width)
    print("Inverted image height:\t%s" % image_height)
    # Find all the lines in the image
    horizontal_lines, vertical_lines = find_lines(inverted_image)

    # show_lines(inverted_image, horizontal_lines)
    # show_lines(inverted_image, vertical_lines)

    def horizontal_sort_func(i):
        return (min(i[0][1], i[0][3]))

    def vertical_sort_func(i):
        return (min(i[0][0], i[0][2]))

    # Now look for the internal coordinates of the matrix cells. Do this first for the horizontal lines,
    # which will give us the y axis coordinates of the cells.
    # 1. Sort by the y value of the line (since the line may not be exactly vertical, sort by the min
    #    y value of the two end points.
    # 2. Since lines are 1 pixel wide, starting at the top, go pixel by pixel. If we have a line that is
    #    the same y value as the previous or is +1, we know we're in the same line in the matrix image
    #    (which are > 1 pixel wide.
    # 3. Otherwise if there is more of a gap, we know we've traversed a cell and are hitting the start
    #    of the line at the other side.
    # 4. In this case, add the coordinates to the list of coordinates.
    horizontal_lines.sort(key=horizontal_sort_func)
    y_coords = []
    for i in range(0, len(horizontal_lines) - 1):
        if horizontal_lines[i + 1][0][1] > horizontal_lines[i][0][1] + 1:
            y_coords.append([horizontal_lines[i][0][1] + 1, horizontal_lines[i + 1][0][1] - 1])
        current = horizontal_lines[i][0][1]
    print('Y coordinates:')
    print(y_coords)

    # Now same for vertical lines and values on the x axis
    vertical_lines.sort(key=vertical_sort_func)
    x_coords = []
    for i in range(0, len(vertical_lines) - 1):
        if vertical_lines[i + 1][0][0] > vertical_lines[i][0][0] + 1:
            x_coords.append([vertical_lines[i][0][0] + 1, vertical_lines[i + 1][0][0] - 1])
        current = vertical_lines[i][0][0]
    print('X coordinates:')
    print(x_coords)

    return x_coords, y_coords