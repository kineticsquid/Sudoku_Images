import cv2
import sys
import json
import util
import requests
import numpy

def main():
    if len(sys.argv) < 3:
        raise Exception("Inputs:\n\tparameter 1: Test image to use for processing."
                        "\n\tparameter 2: file for image metrics json.")
    input_image = sys.argv[1]
    print("Image: %s" % input_image)
    metrics_file = sys.argv[2]
    metrics_file = open(metrics_file, "r")
    training_metrics = json.loads(metrics_file.read())
    puzzle_image = None
    media_url = input_image

    http_headers = {'content-type': 'image/jpg'}
    response = requests.get(media_url, headers=http_headers)
    if response.status_code == 200:
        results = response.content
        image_bytearray = numpy.asarray(bytearray(results), dtype="uint8")
        image = cv2.imdecode(image_bytearray, cv2.IMREAD_GRAYSCALE)
        # util.show_image(image)
        matrix = util.extract_matrix_from_image(image, training_metrics)
    print(matrix)

if __name__ == '__main__':
    main()