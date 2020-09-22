import cv2
import numpy
from matplotlib import pyplot

"""
Routine to redistribute values in an array to an array of different length
"""
def morph_array_to_size(from_array, to_size):
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
    # Now remove blank space on the left
    while x_counts[x_start] == x_min and x_start != x_end:
        x_start += 1
    # If x_start == x_end after removing the line noise from the edges, it means the cell
    # is empty, so return None.
    if x_start == x_end:
        trimmed_image = None
    # Otherwise, finish removing the blank space around the digit.
    else:
        # Remove the blank space on the right
        while x_counts[x_end] == x_min:
            x_end -= 1
        # Now remove blank space on the top
        while y_counts[y_start] == y_min:
            y_start += 1
        # Now remove blank space on the bottom
        while y_counts[y_end] == y_min:
            y_end -= 1
        trimmed_image = raw_image[y_start:y_end, x_start:x_end]

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