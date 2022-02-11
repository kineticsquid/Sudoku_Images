import cv2
import sys
import json
import util
import pytesseract

bw_threshold = 160
image_density_threshold = 0.001

def main():
    if len(sys.argv) < 2:
        raise Exception("Inputs:\n\tparameter 1: Test image to use for processing.")
    input_image = sys.argv[1]
    print("Image: %s" % input_image)
    image = cv2.imread(input_image)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_bw =cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    x_coords, y_coords = util.get_cell_boundaries(img_bw)
    row = 0
    for y_coord in y_coords:
        column = 0
        for x_coord in x_coords:
            raw_image = img_rgb[y_coord[0]:y_coord[1], x_coord[0]:x_coord[1]]
            image_height, image_width = img_bw.shape
            image_sum = raw_image.sum()
            image_density = image_sum / (image_width * image_height * 255)
            if image_density < image_density_threshold or image_density >= 1 - image_density_threshold:
                number = 0
            else:
                custom_oem_psm_config = r'--oem 3 --psm 10'
                number = pytesseract.image_to_string(image, config=custom_oem_psm_config)
                util.show_image(raw_image,
                                title="Row %s, col %s: %s" % (row, column, number))
            print("Row %s\tColumn %s\t%s" % (row, column, number))
            column += 1
        row += 1


if __name__ == '__main__':
    main()