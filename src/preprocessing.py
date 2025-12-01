import numpy as np
from src import utils
import cv2

#Example path
# path = "C:/Users/hp333/Desktop/ocr/Data/Test Data/reverseWaybill-156387426414724544_1.jpg"
# path = ""

def preprocess(img):
    # img = cv2.imread(path)
    # Dynamically sizing
    h, w = img.shape[:2]
    print("height : ", h, " width : ", w)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(gray_img)

    #smoothing
    noise_level = np.std(img)
    print("Noise level: ", noise_level)
    if noise_level > 35:
        img = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
        print("smoothing done")
    elif noise_level > 25:
        img = cv2.GaussianBlur(img, (3, 3), 0)
        print("smoothing done")

    # sharpening using kernels
    laplacian_var = cv2.Laplacian(gray_img, cv2.CV_64F).var()
    print("Laplacian variance:", laplacian_var)

    # Adaptive threshold based on image size
    img_size = w * h
    adaptive_threshold = 100 * (img_size / 1000000)  # Scale with megapixels
    adaptive_threshold = max(50, min(adaptive_threshold, 500))  # Clamp between 50-500

    print(f"Adaptive blur threshold: {adaptive_threshold:.1f}")

    if laplacian_var < adaptive_threshold:
        print("Sharpening applied")
        # Unsharp masking (gentler than direct kernel)
        gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
        img = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)

        # kernel with reduced intensity
        # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) / 1.5
        # img = cv2.filter2D(img, -1, kernel)

    #morphological cleanup
    kernel_size = max(3, min(w, h) // 200)
    opening_kernel = np.ones((max(3, min(w,h)//300), max(3, min(w,h)//300)), np.uint8)
    closing_kernel = np.ones((max(5, min(w,h)//150), max(5, min(w,h)//150)), np.uint8)
    if utils.needs_dilation(img):
        print("dilation done")
        img = cv2.dilate(img, kernel_size)

    if utils.needs_opening(img):
        print("opening done")
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, opening_kernel)

    if utils.needs_closing(img):
        print("closing done")
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, closing_kernel)

    #DPI adjustments
    img = utils.intelligent_dpi_adjustment(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img

    # Results
    # cv2.imshow("preprocessed image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()