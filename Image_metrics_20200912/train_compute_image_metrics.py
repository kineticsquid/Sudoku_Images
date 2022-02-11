"""
Routine to compute metrics for images that are training data for OCR for
numbers. Basically, we extract a rectangle containing a digit, trimmed so there
is no white space around the digit and then calculate the % of pixels that are non-zero at each
point along the x and y axes. Then we compute the average image size. Then compute normalized versions
of these arrays of pixel %s that match the average image width and height.  Finally, we compute the
overall average pixel %s for each digit based on the training images we have.

The final data structure looks like this:

{
  "normalized_size":
    {
        "x": width
        "y": height
    }
  1:
  {
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
  2:
  {
    "raw_data": [],
    "normalized_data": []
  },
  9:
    {
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

def normalize_image_metrics(image_data):
    # Normalize each of the images to the average size. Save the individual normalized image data
    # and save an overall normalized average for each digit.
    normalized_width = image_data['normalized_size']['x']
    normalized_height = image_data['normalized_size']['y']
    for i in range(1, 10):
        for item in image_data[i]['raw_data']:
            new_entry = {}
            new_entry['file_name'] = item['file_name']
            if len(item['x']) == normalized_width:
                new_entry['x'] = item['x']
            else:
                new_entry['x'] = util.morph_array_to_size(item['x'], normalized_width)
            if len(item['y']) == normalized_height:
                new_entry['y'] = item['y']
            else:
                new_entry['y'] = util.morph_array_to_size(item['y'], normalized_height)
            image_data[i]['normalized_data'].append(new_entry)

def main():
    if len(sys.argv) < 3:
        raise Exception("Input parameter 1: folder of folders of training data images. Input parameter 2: folder for JSON output representing image metrics.")
    # input fold of images organized into folders named by the number in in the image
    input_folder = sys.argv[1]
    # output folder for json metrics
    output_folder = sys.argv[2]

    image_data = {}
    for i in range(1,10):
        image_data[i] = {}
        image_data[i]['raw_data'] = []
        image_data[i]['normalized_data'] = []

    # These are to calculate average image width and height
    number_of_items = 0
    width_total = 0
    height_total = 0

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
                        'file_name': image_file,
                        'x': x_percentage.tolist(),
                        'y': y_percentage.tolist()
                    }
                    image_data[current_number]['raw_data'].append(new_entry)

                    number_of_items += 1
                    width_total += len(x_counts)
                    height_total += len(y_counts)

            print("Finished folder: %s" % image_folder)
    normalized_width = int(width_total/number_of_items)
    normalized_height = int(height_total/number_of_items)
    image_data['normalized_size'] = {'x': normalized_width,
                                     'y': normalized_height}
    output_json(image_data, "%s/image_data.json" % output_folder)
    normalize_image_metrics(image_data)

    output_json(image_data, "%s/image_metrics.json" % output_folder)


if __name__ == '__main__':
    main()
