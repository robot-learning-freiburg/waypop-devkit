import collections

import cv2
import numpy as np

FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)

CROSS_KERNEL_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ],
    dtype=np.uint8,
)

CROSS_KERNEL_5 = np.asarray(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ],
    dtype=np.uint8,
)

CROSS_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ],
    dtype=np.uint8,
)


def fill_in_multiscale(
    depth_map,
    max_depth=100.0,
    depth_range=(1.0, 15.0, 30.0),
    dilation_kernel_far=CROSS_KERNEL_3,
    dilation_kernel_med=CROSS_KERNEL_5,
    dilation_kernel_near=CROSS_KERNEL_7,
    extrapolate=False,
    blur_type="bilateral",
    show_process=False,
):
    """Sparse-to-dense depth completion using multi-scale dilation.

    Args:
        depth_map: float array with sparse depth in meters (0 for invalid).
        max_depth: inversion ceiling in meters.
        depth_range: near/med/far bucket edges for dilation strategy.
        extrapolate: extend top values upward (disabled by default).
        blur_type: 'bilateral' or 'gaussian'.
        show_process: whether to return intermediate stages.

    Returns:
        depths_out, process_dict
    """
    depths_in = np.float32(depth_map)

    valid_pixels_near = (depths_in >= depth_range[0]) & (depths_in <=
                                                         depth_range[1])
    valid_pixels_med = (depths_in > depth_range[1]) & (depths_in <=
                                                       depth_range[2])
    valid_pixels_far = depths_in > depth_range[2]

    s1_inverted_depths = np.copy(depths_in)
    valid_pixels = s1_inverted_depths >= depth_range[0]
    s1_inverted_depths[
        valid_pixels] = max_depth - s1_inverted_depths[valid_pixels]

    dilated_far = cv2.dilate(np.multiply(s1_inverted_depths, valid_pixels_far),
                             dilation_kernel_far)
    dilated_med = cv2.dilate(np.multiply(s1_inverted_depths, valid_pixels_med),
                             dilation_kernel_med)
    dilated_near = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_near),
        dilation_kernel_near)

    valid_pixels_near = dilated_near > 0.1
    valid_pixels_med = dilated_med > 0.1
    valid_pixels_far = dilated_far > 0.1

    s2_dilated_depths = np.copy(s1_inverted_depths)
    s2_dilated_depths[valid_pixels_far] = dilated_far[valid_pixels_far]
    s2_dilated_depths[valid_pixels_med] = dilated_med[valid_pixels_med]
    s2_dilated_depths[valid_pixels_near] = dilated_near[valid_pixels_near]

    s3_closed_depths = cv2.morphologyEx(s2_dilated_depths, cv2.MORPH_CLOSE,
                                        FULL_KERNEL_5)

    s4_blurred_depths = np.copy(s3_closed_depths)
    blurred = cv2.medianBlur(s3_closed_depths, 5)
    valid_pixels = s3_closed_depths > 0.1
    s4_blurred_depths[valid_pixels] = blurred[valid_pixels]

    top_mask = np.ones(depths_in.shape, dtype=bool)
    for pixel_col_idx in range(s4_blurred_depths.shape[1]):
        pixel_col = s4_blurred_depths[:, pixel_col_idx]
        top_pixel_row = np.argmax(pixel_col > 0.1)
        top_mask[0:top_pixel_row, pixel_col_idx] = False

    valid_pixels = s4_blurred_depths > 0.1
    empty_pixels = ~valid_pixels & top_mask

    dilated = cv2.dilate(s4_blurred_depths, FULL_KERNEL_9)
    s5_dilated_depths = np.copy(s4_blurred_depths)
    s5_dilated_depths[empty_pixels] = dilated[empty_pixels]

    s6_extended_depths = np.copy(s5_dilated_depths)
    top_mask = np.ones(s5_dilated_depths.shape, dtype=bool)
    top_row_pixels = np.argmax(s5_dilated_depths > 0.1, axis=0)
    top_pixel_values = s5_dilated_depths[top_row_pixels,
                                         range(s5_dilated_depths.shape[1])]

    for pixel_col_idx in range(s5_dilated_depths.shape[1]):
        if extrapolate:
            s6_extended_depths[0:top_row_pixels[pixel_col_idx],
                               pixel_col_idx] = top_pixel_values[pixel_col_idx]
        else:
            top_mask[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = False

    s7_blurred_depths = np.copy(s6_extended_depths)
    for _ in range(6):
        empty_pixels = (s7_blurred_depths < 0.1) & top_mask
        dilated = cv2.dilate(s7_blurred_depths, FULL_KERNEL_5)
        s7_blurred_depths[empty_pixels] = dilated[empty_pixels]

    blurred = cv2.medianBlur(s7_blurred_depths, 5)
    valid_pixels = (s7_blurred_depths > 0.1) & top_mask
    s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    if blur_type == "gaussian":
        blurred = cv2.GaussianBlur(s7_blurred_depths, (5, 5), 0)
        valid_pixels = (s7_blurred_depths > 0.1) & top_mask
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]
    elif blur_type == "bilateral":
        blurred = cv2.bilateralFilter(s7_blurred_depths, 5, 0.5, 2.0)
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    s8_inverted_depths = np.copy(s7_blurred_depths)
    valid_pixels = np.where(s8_inverted_depths > 0.1)
    s8_inverted_depths[
        valid_pixels] = max_depth - s8_inverted_depths[valid_pixels]
    depths_out = s8_inverted_depths

    process_dict = None
    if show_process:
        process_dict = collections.OrderedDict()
        process_dict["s0_depths_in"] = depths_in
        process_dict["s1_inverted_depths"] = s1_inverted_depths
        process_dict["s2_dilated_depths"] = s2_dilated_depths
        process_dict["s3_closed_depths"] = s3_closed_depths
        process_dict["s4_blurred_depths"] = s4_blurred_depths
        process_dict["s5_combined_depths"] = s5_dilated_depths
        process_dict["s6_extended_depths"] = s6_extended_depths
        process_dict["s7_blurred_depths"] = s7_blurred_depths
        process_dict["s8_inverted_depths"] = s8_inverted_depths
        process_dict["s9_depths_out"] = depths_out

    return depths_out, process_dict
