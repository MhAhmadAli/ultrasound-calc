import cv2

def crop_bottom_pixels(image, num_pixels=1):
    img_height = image.shape[0]

    end_row = img_height - num_pixels
    start_row = 0

    cropped_image = image[start_row:end_row, :]
    return cropped_image

image_path = "image.png"
image = cv2.imread(image_path)
cropped_image = crop_bottom_pixels(image)

cv2.imwrite("cropped_image.png", cropped_image)

# cv2.imshow("original image", image)
# cv2.imshow("cropped image", cropped_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

