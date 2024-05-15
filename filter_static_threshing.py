import cv2
import numpy as np
import pandas as pd
import os

def filter_static_threshing(image):
    thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return opening

def get_contours(opening):
    invert_opening = np.invert(opening)
    contours, _ = cv2.findContours(invert_opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_image = cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

    return (contours, contour_image)

def process_image(image):
    # Initialize variables to store the lowest and highest points
    lowest_point = (0, 0)
    highest_point = (0, 0)

    # Apply the filter_static_threshing function
    opening = filter_static_threshing(image)

    # Get the contours and the contour image
    contours, contour_image = get_contours(opening)

    # Iterate through each contour (peak)
    for contour in contours:
        # Find the top and bottom points of the contour
        top_point = tuple(contour[contour[:,:,1].argmin()][0])
        bottom_point = tuple(contour[contour[:,:,1].argmax()][0])

        # Update lowest and highest points if necessary
        if top_point[1] > highest_point[1]:
            highest_point = top_point
        if bottom_point[1] < lowest_point[1] or lowest_point == (0, 0):
            lowest_point = bottom_point

    cv2.drawKeypoints(contour_image, [cv2.KeyPoint(float(highest_point[0]), float(highest_point[1]), 5)], contour_image, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.drawKeypoints(contour_image, [cv2.KeyPoint(float(lowest_point[0]),float(lowest_point[1]), 5)], contour_image, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('contour image', contour_image)
    cv2.imshow('original', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Convert lowest and highest points to the scale from 0 to 200
    scale_factor = 200 / image.shape[0]
    lowest_point_scaled = (lowest_point[0], int(lowest_point[1] * scale_factor))
    highest_point_scaled = (highest_point[0], int(highest_point[1] * scale_factor))

    print(f'Lowest point: {lowest_point_scaled}')
    print(f'Highest point: {highest_point_scaled}')
    exit()

def main():
    output = pd.DataFrame()
    # walk through the image folder
    # for each image, apply the filter_static_threshing function
    for root, dirs, files in os.walk('images'):
        for file in files:
            if(file.endswith(".png")):
                print(f'Processing {os.path.join(root, file)}')
                image = cv2.imread(os.path.join(root, file), cv2.IMREAD_GRAYSCALE)
                process_image(image)

    # cv2.imshow('original', image)
    # cv2.imshow('invert image', invert_opening)
    # cv2.imshow('contour image', contour_image)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
