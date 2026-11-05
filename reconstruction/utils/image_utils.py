import numpy as np
from PIL import Image, ImageFilter, ImageChops, ImageEnhance
from skimage.color import rgba2rgb, rgb2lab, lab2rgb
from skimage import exposure
from skimage import data
from skimage import exposure
from skimage.exposure import match_histograms
import cv2

SEGMENTATION_COLOR_MAP = {
    0: [0, 0, 0],
    1: [102, 102, 102],
    2: [0, 0, 142],
    3: [0, 0, 70],
    4: [0, 60, 100],
    5: [61, 133, 198],
    6: [119, 11, 32],
    7: [0, 0, 230],
    8: [111, 168, 220],
    9: [220, 20, 60],
    10: [255, 0, 0],
    11: [180, 0, 0],
    12: [127, 96, 0],
    13: [91, 15, 0],
    14: [230, 145, 56],
    15: [153, 153, 153],
    16: [234, 153, 153],
    17: [246, 178, 107],
    18: [250, 170, 30],
    19: [70, 70, 70],
    20: [128, 64, 128],
    21: [234, 209, 220],
    22: [217, 210, 233],
    23: [244, 35, 232],
    24: [107, 142, 35],
    25: [70, 130, 180],
    26: [102, 102, 102],
    27: [102, 102, 102],
    28: [102, 102, 102],
    29: [0, 255, 255],
    30: [255, 255, 255]
}


def adjust_brightness(rgba_image_array, reference_rgb_image_array):
    """Adjust the brightness of the input image towards the brightness of the reference"""

    non_zero_alpha_pixels = rgba_image_array[:, :, 3] > 0
    rgba_mean_brightness = np.mean(rgba_image_array[non_zero_alpha_pixels, :3])
    reference_mean_brightness = np.mean(reference_rgb_image_array[:, :, :3])
    brightness_factor = reference_mean_brightness / rgba_mean_brightness
    rgba_image_array = rgba_image_array.astype(np.float64)
    rgba_image_array[:, :, :3] *= brightness_factor
    return rgba_image_array.astype(np.uint8)


def histogram_match_with_reference(image_array, reference_array):
    """Adjust the histogram of input image to match the histogram of the reference image.
    Args:
        image_array: np.array of shape (n, m, 4) input image RGBA
        reference_array: np.array of shape (o, p, 3) reference image RGB"""

    if reference_array.shape[2] != 3:
        raise ValueError("Reference array must be in RGB format.")

    alpha_channel = image_array[:, :, 3] / 255.0

    reference_lab = rgb2lab(reference_array)

    non_zero_alpha_pixels = image_array[:, :, 3] > 0

    for channel in range(3):
        flat_channel = image_array[non_zero_alpha_pixels, :3][:,
                                                              channel].astype(
                                                                  np.uint8)
        reference_channel = reference_lab[:, :,
                                          channel].flatten().astype(np.uint8)
        matched_channel = exposure.match_histograms(flat_channel,
                                                    reference_channel)
        image_array[
            non_zero_alpha_pixels, :3][:, channel] = matched_channel.reshape(
                flat_channel.shape)

    modified_image = np.dstack(
        (image_array.astype(np.uint8), alpha_channel.astype(np.uint8)))

    return image_array.astype(np.uint8)


def histogram_match_with_soft_blend(image_array,
                                    reference_array,
                                    alpha_blend_strength=1.0):
    """
    Matches the L*a*b* histogram of an RGBA image to an RGB reference image.

    The luminance (L*) channel is transformed to match the mean and standard
    deviation of the reference, and the result is blended with the original
    luminance based on the input image's alpha channel. The a* and b* channels
    are preserved to maintain the original color relationships.

    Args:
        image_array (np.ndarray): The input image as an RGBA NumPy array.
        reference_array (np.ndarray): The reference image as an RGB NumPy array.
        alpha_blend_strength (float): A value between 0.0 and 1.0 that controls
                                      the strength of the alpha-based blending.
                                      1.0 means the alpha channel fully
                                      determines the blending; 0.0 means
                                      no blending occurs.

    Returns:
        np.ndarray: The histogram-matched image with an alpha channel as an
                    RGBA NumPy array.
    """

    if reference_array.ndim != 3 or reference_array.shape[2] != 3:
        print(reference_array.shape)
        raise ValueError(
            "Reference image must be an RGB image with shape (H, W, 3).")

    if image_array.ndim != 3 or image_array.shape[2] != 4:
        raise ValueError(
            "Input image must be an RGBA image with shape (H, W, 4).")

    rgb = image_array[..., :3].astype(np.float64) / 255.0
    alpha = image_array[..., 3].astype(np.float64) / 255.0
    ref_rgb = reference_array.astype(np.float64) / 255.0

    img_lab = rgb2lab(rgb)
    ref_lab = rgb2lab(ref_rgb)

    L_src = img_lab[..., 0]
    L_ref = ref_lab[..., 0]

    mask = alpha > 1e-6
    L_src_masked = L_src[mask]

    if not L_src_masked.size:
        return image_array

    L_ref_flat = L_ref.flatten()

    mean_src = np.mean(L_src_masked)
    std_src = np.std(L_src_masked)
    mean_ref = np.mean(L_ref_flat)
    std_ref = np.std(L_ref_flat)

    if std_src < 1e-6:
        std_src = 1e-6

    L_adjusted = (L_src - mean_src) * (std_ref / std_src) + mean_ref

    low_L_threshold = 20
    high_L_threshold = 50

    preservation_weight = np.zeros_like(L_src)

    idx_very_dark = L_src < low_L_threshold
    preservation_weight[idx_very_dark] = 1.0

    idx_transition = (L_src >= low_L_threshold) & (L_src < high_L_threshold)
    preservation_weight[idx_transition] = 1.0 - (
        (L_src[idx_transition] - low_L_threshold) /
        (high_L_threshold - low_L_threshold))

    L_adjusted_blended = (preservation_weight * L_src) + (
        (1 - preservation_weight) * L_adjusted)

    effective_alpha = alpha * alpha_blend_strength

    L_final_blended = (
        1 - effective_alpha) * L_src + effective_alpha * L_adjusted_blended

    output_lab = img_lab.copy()
    output_lab[..., 0] = L_final_blended

    matched_rgb = lab2rgb(output_lab) * 255
    matched_rgb = matched_rgb.clip(0, 255).astype(np.uint8)

    return np.dstack((matched_rgb, (alpha * 255).astype(np.uint8)))


def blend_image_with_shadow(image_array, segmentation_mask, shadow_mask,
                            reference_array):
    """
    Blends an object (RGB) with a shadow into a BGR reference image using both a
    segmentation mask for the object and a separate mask for the shadow.
    This version uses a direct luminance adjustment to preserve the object's original color.

    Args:
        image_array (np.ndarray): The foreground object as an RGB array (h, w, 3).
        segmentation_mask (np.ndarray): The segmentation mask for the object (h, w).
        shadow_mask (np.ndarray): The binary mask for the shadow (h, w).
        reference_array (np.ndarray): The background image as a BGR array (H, W, 3).

    Returns:
        np.ndarray: The blended image as a BGR array (H, W, 3).
    """

    if np.max(segmentation_mask) == 0:
        print(
            "Warning: The segmentation mask appears to be all zeros. The object will be invisible."
        )

    if image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]

    h, w, _ = image_array.shape
    H, W, _ = reference_array.shape

    if H < h or W < w:
        print(
            "Warning: Foreground image is larger than background. Cropping foreground and masks."
        )
        h, w = min(h, H), min(w, W)
        image_array = image_array[:h, :w]
        segmentation_mask = segmentation_mask[:h, :w]
        shadow_mask = shadow_mask[:h, :w]

    fg_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    bg_gray = cv2.cvtColor(reference_array, cv2.COLOR_BGR2GRAY)

    object_luminance = np.mean(fg_gray[segmentation_mask > 0])
    background_luminance = np.mean(bg_gray)

    luminance_factor = background_luminance / (object_luminance + 1e-6)

    transferred_image_array = (image_array.astype(np.float32) *
                               luminance_factor).clip(0, 255).astype(np.uint8)

    blended_image = reference_array.copy()

    shadow_alpha = shadow_mask.astype(np.float32) / 255.0
    shadow_alpha_3ch = np.stack([shadow_alpha] * 3, axis=-1)

    shadow_roi = blended_image[0:h, 0:w]

    dark_bg_roi = shadow_roi * 0.2

    blended_shadow_roi = (dark_bg_roi *
                          shadow_alpha_3ch) + (shadow_roi *
                                               (1 - shadow_alpha_3ch))

    blended_image[0:h, 0:w] = blended_shadow_roi

    object_alpha = segmentation_mask.astype(np.float32) / 255.0
    object_alpha_3ch = np.stack([object_alpha] * 3, axis=-1)

    bgr_image = cv2.cvtColor(transferred_image_array, cv2.COLOR_RGB2BGR)

    object_roi = blended_image[0:h, 0:w]

    final_roi = (bgr_image * object_alpha_3ch) + (object_roi *
                                                  (1 - object_alpha_3ch))

    blended_image[0:h, 0:w] = final_roi

    return blended_image


def blend_image_with_shadow(image_array, segmentation_mask, shadow_mask,
                            reference_array):
    """
    Blends an object (RGB) with a shadow into a BGR reference image using both a
    segmentation mask for the object and a separate mask for the shadow.

    Args:
        image_array (np.ndarray): The foreground object as an RGB array (h, w, 3).
        segmentation_mask (np.ndarray): The segmentation mask for the object (h, w).
        shadow_mask (np.ndarray): The binary mask for the shadow (h, w).
        reference_array (np.ndarray): The background image as a BGR array (H, W, 3).

    Returns:
        np.ndarray: The blended image as a BGR array (H, W, 3).
    """

    if np.max(segmentation_mask) == 0:
        print(
            "Warning: The segmentation mask appears to be all zeros. The object will be invisible."
        )

    if image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]

    h, w, _ = image_array.shape
    H, W, _ = reference_array.shape

    if H < h or W < w:
        print(
            "Warning: Foreground image is larger than background. Cropping foreground and masks."
        )
        h, w = min(h, H), min(w, W)
        image_array = image_array[:h, :w]
        segmentation_mask = segmentation_mask[:h, :w]
        shadow_mask = shadow_mask[:h, :w]

    sample_size = 100
    if H > sample_size and W > sample_size:
        x = np.random.randint(0, H - sample_size)
        y = np.random.randint(0, W - sample_size)
        background_sample = reference_array[x:x + sample_size,
                                            y:y + sample_size, :]
    else:
        background_sample = reference_array

    fg_lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
    bg_lab = cv2.cvtColor(background_sample, cv2.COLOR_BGR2LAB)

    fg_mean, fg_std = cv2.meanStdDev(fg_lab)
    bg_mean, bg_std = cv2.meanStdDev(bg_lab)

    transferred_lab = fg_lab.copy()
    for i in range(3):
        transferred_lab[:, :,
                        i] = ((fg_lab[:, :, i] - fg_mean[i]) *
                              (bg_std[i] / (fg_std[i] + 1e-6))) + bg_mean[i]

    transferred_image_array = cv2.cvtColor(transferred_lab.astype(np.uint8),
                                           cv2.COLOR_LAB2RGB)

    blended_image = reference_array.copy()

    shadow_alpha = shadow_mask.astype(np.float32) / 255.0
    shadow_alpha_3ch = np.stack([shadow_alpha] * 3, axis=-1)

    shadow_roi = blended_image[0:h, 0:w]

    dark_bg_roi = shadow_roi * 0.2

    blended_shadow_roi = (dark_bg_roi *
                          shadow_alpha_3ch) + (shadow_roi *
                                               (1 - shadow_alpha_3ch))

    blended_image[0:h, 0:w] = blended_shadow_roi

    object_alpha = segmentation_mask.astype(np.float32) / 255.0
    object_alpha_3ch = np.stack([object_alpha] * 3, axis=-1)

    bgr_image = cv2.cvtColor(transferred_image_array, cv2.COLOR_RGB2BGR)

    object_roi = blended_image[0:h, 0:w]

    final_roi = (bgr_image * object_alpha_3ch) + (object_roi *
                                                  (1 - object_alpha_3ch))

    blended_image[0:h, 0:w] = final_roi

    return blended_image


def smooth_alpha(img_array, radius=2):
    """Smooth the alpha channel of the input array using Gaussian Blur"""
    target_img = Image.fromarray(img_array)
    alpha = target_img.split()[3]
    alpha_smoothed = alpha.filter(ImageFilter.GaussianBlur(radius=radius))
    alpha_smoothed_twice = alpha_smoothed.filter(
        ImageFilter.GaussianBlur(radius=2 * radius))
    target_img.putalpha(alpha_smoothed)

    return np.array(target_img)


def histogram_match(image_array, reference_array):

    reference_alpha = np.ones(
        (reference_array.shape[0], reference_array.shape[1], 1),
        dtype=np.uint8) * 255
    reference_with_alpha = np.concatenate((reference_array, reference_alpha),
                                          axis=-1)
    target = np.copy(reference_with_alpha)
    alpha_mask = image_array[:, :, 3] > 0
    target[alpha_mask] = image_array[alpha_mask]

    matched_target = match_histograms(target,
                                      reference_with_alpha,
                                      channel_axis=-1)
    Image.fromarray(matched_target).save(
        "Presentation_Demo/matched_target_.png")

    transparent_result = np.zeros(np.shape(image_array))
    transparent_result[alpha_mask] = matched_target[alpha_mask]
    final_image = match_histograms(np.array(image_array, dtype=np.uint8),
                                   np.array(transparent_result,
                                            dtype=np.uint8),
                                   channel_axis=-1)

    return final_image
