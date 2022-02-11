import cv2
import sys
import json
import util

def main():
    if len(sys.argv) < 3:
        raise Exception("Inputs:\n\tparameter 1: Test image to use for processing."
                        "\n\tparameter 2: file for image metrics json.")
    input_image = sys.argv[1]
    print("Image: %s" % input_image)
    metrics_file = sys.argv[2]
    metrics_file = open(metrics_file, "r")
    training_metrics = json.loads(metrics_file.read())
    image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    matrix = util.extract_matrix_from_image(image, training_metrics)
    print(matrix)

if __name__ == '__main__':
    main()