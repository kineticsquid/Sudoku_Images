"""
Routine to compute metrics for images that are training data for OCR for
numbers. Basically, we extract a rectangle containing a digit, trimmed so there
is no white space around the digit and then calculate the % of pixels that are non-zero at each
point along the x and y axes. Then we compute the average image size. Then compute normalized versions
of these arrays of pixel %s that match the average image width and height.  Finally, we compute the
overall average pixel %s for each digit based on the training images we have.

The final data structure looks like this:

{
  "size":
  {
    "x": "average_width",
    "y": "average_height"
  },
  "1":
  {
    "average":
    {
      "x": [0, 1, 2, "average_width"],
      "y": [0, 1 ,2, "average_height"]
    },
    "raw_data":
    [
      {
        "filename": filename,
        "x": [0, 1 ,2, "width"],
        "y": [0, 1, 2, "height"]
      },
      {
        "filename": filename,
        "x": [],
        "y": []
      }
    ],
    "normalized_data":
    [
      {
        "x": [0, 1 ,2, "average_width"],
        "y": [0, 1, 2, "average_height"]
      },
      {
        "x": [],
        "y": []
      }
    ]
  },
  "2":
  {
    "average": {},
    "raw_data": [],
    "normalized_data": []
  },
  "9":
    {
    "average": {},
    "raw_data": [],
    "normalized_data": []
  }
}

"""
import cv2
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

def output_json(json_to_output, output_file_name):
    print(json.dumps(json_to_output, indent=4))
    output_file = open(output_file_name, 'w')
    output_file.write(json.dumps(json_to_output, indent=4))
    output_file.close()

def calculate_image_metrics(raw_image_data):
    image_metrics = {}

    # First, calculate average image width and height
    # Also begin to set up the final output JSON including copying the raw image data
    number_of_items = 0
    width_total = 0
    height_total = 0
    for i in range(1, 10):
        index = str(i)
        image_metrics[index] = {}
        image_metrics[index]['raw_data'] = []
        for item in raw_image_data[index]:
            number_of_items += 1
            width_total += len(item['x'])
            height_total += len(item['y'])
            image_metrics[index]['raw_data'].append(item)

    # Add the overall average image dimensions
    average_width = int(width_total/number_of_items)
    average_height = int(height_total/number_of_items)
    image_metrics['size'] = {}
    image_metrics['size']['x'] = average_width
    image_metrics['size']['y'] = average_height

    # Now normalize each of the images to the average size. Save the individual normalized image data
    # and save an overall normalized average for each digit.
    for i in range(1, 10):
        index = str(i)
        image_metrics[index]['normalized_data'] = []
        x_total = numpy.zeros((average_width), dtype=float)
        y_total = numpy.zeros((average_height), dtype=float)
        for item in image_metrics[index]:
            new_entry = {}
            if len(item['x']) == average_width:
                new_entry['x'] = item['x']
            else:
                new_entry['x'] = util.morph_array_to_size(item['x'], average_width)
            if len(item['y']) == average_height:
                new_entry['y'] = item['y']
            else:
                new_entry['y'] = util.morph_array_to_size(item['y'], average_height)
            x_total += new_entry['x']
            y_total += new_entry['y']
            image_metrics[index]['normalized_data'].append(new_entry)
        x_average = x_total / len(image_metrics[index])
        y_average = y_total / len(image_metrics[index])
        image_metrics[index]['average'] = {}
        image_metrics[index]['average']['x'] = x_average
        image_metrics[index]['average']['y'] = y_average

    return image_metrics

def main():
    if len(sys.argv) < 3:
        raise Exception("Input parameter 1: folder of folders of training data images. Input parameter 2: folder for JSON output representing image metrics.")
    # input fold of images organized into folders named by the number in in the image
    input_folder = sys.argv[1]
    # output folder for json metrics
    output_folder = sys.argv[2]

    raw_image_data = {}
    for i in range(1,10):
        raw_image_data[str(i)] = []

    for image_folder_name in listdir(input_folder):
        image_folder = "%s/%s" % (input_folder, image_folder_name)
        if os.path.isdir(image_folder):
            print("Processing: %s..." % image_folder_name)
            # current_number is the number in the images in this folder
            current_number = int(image_folder_name)
            for image_file in listdir(image_folder):
                image_file_name, image_file_extension = os.path.splitext(image_file)
                if image_file_extension == '' or image_file_name[0] == '.':
                    print("Skipping, not an image file.")
                else:
                    image = cv2.imread("%s/%s" % (image_folder, image_file), cv2.IMREAD_GRAYSCALE)
                    trimmed_image = util.trim(image)
                    # This is summing columns vertically
                    x_counts = trimmed_image.sum(axis=0)
                    # This is summing rows horizontally
                    y_counts = trimmed_image.sum(axis=1)
                    image_size = len(x_counts) * len(y_counts)
                    x_percentage = x_counts / image_size * 100
                    y_percentage = y_counts / image_size * 100

                    new_entry = {
                        'filename': image_file,
                        'x': x_percentage.tolist(),
                        'y': y_percentage.tolist()
                    }
                    raw_image_data[image_folder_name].append(new_entry)

            print("Finished folder: %s" % image_folder)
    output_json(raw_image_data, "%s/raw_image_data.json" % output_folder)
    # :todo start here checking the calcuate metrics routine
    image_metrics = calculate_image_metrics(raw_image_data)

    output_json(image_metrics, "%s/json_to_output.json" % output_folder)


if __name__ == '__main__':
    main()
