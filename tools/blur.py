import cv2
import numpy as np

def crop_img_circle_box(img, left_x, top_y, height, width, radius):
    roi = img[top_y:top_y + height, left_x:left_x + width]
    if roi.size == 0:
        raise ValueError('crop area is out of image bounds')

    h, w = roi.shape[:2]
    radius = max(0, min(radius, h // 2, w // 2))
    roi_mask = np.zeros((h, w), dtype=np.uint8)

    cv2.rectangle(roi_mask, (radius, 0), (w - radius, h), 255, -1)
    cv2.rectangle(roi_mask, (0, radius), (w, h - radius), 255, -1)
    for center in ((radius, radius), (w - radius - 1, radius), (radius, h - radius - 1), (w - radius - 1, h - radius - 1)):
        cv2.circle(roi_mask, center, radius, 255, -1)

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[top_y:top_y + h, left_x:left_x + w] = roi_mask
    retangle_mask_without_circle = np.zeros(img.shape[:2], dtype=np.uint8)

    # retangle_mask_without_circle[top_y:top_y + h, left_x:left_x + w] = 255
    cropped_img = cv2.bitwise_and(img, img, mask=mask)
    mask = np.repeat(mask[..., np.newaxis], 3, axis=2)
    return cropped_img, mask

def gaussian_blur(img, blur_radius, mask = None):
    blur_radius = max(0, int(blur_radius))
    if blur_radius == 0:
        return cv2.bitwise_and(img, img, mask=mask) if mask is not None else img

    kernel_size = blur_radius * 2 + 1
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return cv2.bitwise_and(blurred, mask) if mask is not None else blurred



def change_brightness(image, value):
    value = int(value)
    if value == 0:
        return image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_new = cv2.add(v, value)
    final_hsv = cv2.merge((h, s, v_new))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    
    return img

def change_brightness(image, value):
    image = np.power(image / 255.0, -value + 1)
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image


def combine(img, cropped_img, mask):
    new_img = img.copy()
    mask = mask.astype(bool)
    print(new_img.shape, cropped_img.shape, mask.shape)
    new_img[mask] = cropped_img[mask]
    return new_img




left_x = 320
top_y = 136
height = 810
width = 1280
radius = 30
blur_radius = 100
brightness = -0.9


# left_x = 1300
# top_y = 136
# height = 810
# width = 550
# radius = 30
# blur_radius = 70
# brightness = -0.6

img_path = '192110.jpg'
img_path = 'scene.png'
img_path = 'yi.jpg'
img = cv2.imread(img_path)
# img = cv2.resize(img, (1920, 1080))
if img is None:
    raise FileNotFoundError(img_path)


new_img = img.copy()
new_img = gaussian_blur(new_img, blur_radius)
new_img = change_brightness(new_img, brightness)
cropped_img, mask = crop_img_circle_box(new_img, left_x, top_y, height, width, radius)
new_img = combine(img, cropped_img, mask)
cv2.imwrite('cropped_img.png', new_img)
cv2.imwrite('mask.png', mask)




