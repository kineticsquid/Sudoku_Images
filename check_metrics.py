import sys
import json

def main():
    if len(sys.argv) < 2:
        raise Exception("Need name of metrics file.")
    metrics_file = sys.argv[1]
    metrics_file = open(metrics_file, "r")
    training_metrics = json.loads(metrics_file.read())
    image_width = training_metrics['normalized_size']['x']
    image_height = training_metrics['normalized_size']['y']

    print("Image width:\t%s\t\tImage height:\t%s" % (image_width, image_height))
    for n in range(1, 10):
        for entry in training_metrics[str(n)]['normalized_data']:
            if len(entry['x']) != image_width:
                print("%s x:\t%s" % (entry['file_name'], len(entry['x'])))
            if len(entry['y']) != image_height:
                print("%s y:\t%s" % (entry['file_name'], len(entry['y'])))


if __name__ == '__main__':
    main()