import cv2
import numpy as np
from PIL import Image


def auto_deskew(image_path, out_path=None):
    """
    Auto-detects skew angle and rotates the image to correct it.
    Good for mild tilts (< ~45 degrees).
    Saves corrected image to out_path, or overwrites the original if None.
    Returns the corrected image path.
    """
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(
        dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    angles = []
    for c in contours:
        if cv2.contourArea(c) < 500:
            continue
        rect = cv2.minAreaRect(c)
        angle = rect[-1]
        if angle < -45:
            angle = 90 + angle
        angles.append(angle)

    if not angles:
        print("No skew detected, returning original image.")
        return str(image_path)

    skew_angle = float(np.median(angles))
    print(f"Detected skew angle: {skew_angle:.2f} degrees")

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
    corrected = cv2.warpAffine(img, M, (w, h),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)

    save_path = out_path if out_path else str(image_path)
    cv2.imwrite(str(save_path), corrected)
    print(f"Saved rectified image: {save_path}")
    return str(save_path)


def manual_deskew(image_path, angle_degrees, out_path=None):
    """
    Rotates the image by a known angle.
    Use negative values for counter-clockwise, positive for clockwise.
    """
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -angle_degrees, 1.0)
    corrected = cv2.warpAffine(img, M, (w, h),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)

    save_path = out_path if out_path else str(image_path)
    cv2.imwrite(str(save_path), corrected)
    print(f"Saved rectified image: {save_path}")
    return str(save_path)
