import cv2
import numpy as np
import pandas as pd
import math
import os

def map_value(in_val, in_min, in_max, out_min, out_max):
    return (in_val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def crop_bottom_pixels(image, num_pixels=1):
    img_height = image.shape[0]

    end_row = img_height - num_pixels
    start_row = 0

    cropped_image = image[start_row:end_row, :]
    return cropped_image

def filter_static_threshing(image):
    thresh = cv2.threshold(image, 100, 150, cv2.THRESH_BINARY)[1]

    thresh = cv2.bitwise_not(thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    for (index_rows, rows) in enumerate(opening):
        for (index_pixel, pixel) in enumerate(rows):
            if pixel < 150:
                opening[index_rows][index_pixel] = 0

    return opening

def get_contours(opening):
    invert_opening = np.invert(opening)
    contours, _ = cv2.findContours(invert_opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_image = cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)

    return (contours, contour_image)

def process_image(image):
    image_height = image.shape[0]

    # Apply the filter_static_threshing function
    opening = filter_static_threshing(image)
    # Get the contours and the contour image
    contours, contour_image = get_contours(opening)
                          
    # Initialize variables to store lowest and highest points
    lowest_point = None
    highest_point = None

    # Iterate through contours
    for contour in contours:
        # Iterate through points in the contour
        for point in contour:
            # Extract x and y coordinates
            x, y = point[0]
            # Check if the current point is lower than the lowest_point
            if lowest_point is None or y > lowest_point[1]:
                lowest_point = (x, y)
            # Check if the current point is higher than the highest_point
            if highest_point is None or y < highest_point[1]:
                highest_point = (x, y)

    lowest_value = 0
    highest_value = 0
    
    if lowest_point is not None and highest_point is not None:
        lowest_value = map_value(lowest_point[1], 0, image_height, 200, 0)
        highest_value = map_value(highest_point[1], 0, image_height, 200, 0)

    lowest_value = math.ceil(lowest_value)
    highest_value = math.ceil(highest_value)

    return (lowest_value, highest_value)

def main():
    output = {
        'imageFullPath': [],
        'imageId': [],
        'lowest_point': [],
        'highest_point': []
    }
    # walk through the image folder
    # for each image, apply the filter_static_threshing function
    for root, dirs, files in os.walk('images'):
        for file in files:
            if(file.endswith(".png")):
                print(f'Processing {file}')
                image = cv2.imread(os.path.join(root, file), cv2.IMREAD_GRAYSCALE)
                cropped_image = crop_bottom_pixels(image)
                (lowest_point, highest_point) = process_image(cropped_image)
                output['imageFullPath'].append(os.path.join(root, file))
                output['imageId'].append(file.split('_')[0])
                output['lowest_point'].append(lowest_point)
                output['highest_point'].append(highest_point)

    output_df = pd.DataFrame(output)
    output_df.to_excel('out.xlsx')


if __name__ == '__main__':
    main()
