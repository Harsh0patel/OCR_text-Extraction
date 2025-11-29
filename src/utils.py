import cv2
import numpy as np

def needs_opening(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    if len(np.unique(gray)) == 2:  # Already binary
        bw = gray
    else:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if num_labels < 2:
        return False

    img_area = bw.shape[0] * bw.shape[1]

    small_objects = sum(
        1 for area in stats[1:, cv2.CC_STAT_AREA]  # Skip background (index 0)
        if area < (img_area * 0.0001)  # Less than 0.01% of image
    )
    noise_threshold = max(20, num_labels * 0.1)  # At least 20 or 10% of objects
    return small_objects > noise_threshold


def needs_closing(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    edges = cv2.Canny(gray, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    edge_density = np.count_nonzero(edges) / edges.size
    grad_std = np.std(grad)

    return edge_density > 0.15 or grad_std > 40


def needs_dilation(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    if len(np.unique(gray)) == 2:
        bw = gray
    else:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.mean(bw) > 127:
        bw = cv2.bitwise_not(bw)
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
    text_pixels = dist[dist > 0]

    if len(text_pixels) == 0:  # No text found
        return False

    mean_stroke = np.mean(text_pixels)
    median_stroke = np.median(text_pixels)
    min_stroke = max(1.5, min(bw.shape) * 0.003)
    return mean_stroke < min_stroke or median_stroke < 1.0

def intelligent_dpi_adjustment(img):
    """
    Complete intelligent DPI adjustment for OCR.
    Combines multiple strategies for best results.
    """
    h, w = img.shape[:2]
    print(f"\n{'='*50}")
    print(f"Original dimensions: {w}x{h}")

    # Strategy 1: Check if image is reasonable size
    min_dimension = min(w, h)
    max_dimension = max(w, h)

    # Rule 1: Minimum size check (for very small images)
    if max_dimension < 800:
        scale = 800 / max_dimension
        print(f"Rule: Image too small, scaling up by {scale:.2f}x")

    # Rule 2: Maximum size check (for very large images)
    elif max_dimension > 4000:
        scale = 4000 / max_dimension
        print(f"Rule: Image too large, scaling down by {scale:.2f}x")

    # Rule 3: Optimal range (1200-2500px on longest edge)
    elif max_dimension < 1200:
        scale = 1500 / max_dimension
        print(f"Rule: Upscaling to optimal range ({scale:.2f}x)")
    elif max_dimension > 2500:
        scale = 2000 / max_dimension
        print(f"Rule: Downscaling to optimal range ({scale:.2f}x)")
    else:
        scale = 1.0
        print(f"Rule: Size already optimal, no scaling needed")

    # Apply scaling
    if scale != 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Choose best interpolation
        if scale > 1.0:
            interpolation = cv2.INTER_CUBIC  # Smooth upscaling
            print("Interpolation: INTER_CUBIC (upscaling)")
        else:
            interpolation = cv2.INTER_AREA  # Sharp downscaling
            print("Interpolation: INTER_AREA (downscaling)")

        img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
        print(f"Final dimensions: {new_w}x{new_h}")
        print(f"Equivalent DPI: ~{new_h / 11 * 300:.0f} DPI (assuming 11\" height)")

    print(f"{'='*50}\n")
    return img