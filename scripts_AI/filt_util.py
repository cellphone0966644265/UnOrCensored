
import numpy as np

def medfilt(data, window):
    """Apply a median filter to a 1D array."""
    if window % 2 == 0 or window < 0:
        raise ValueError("Median filter window must be an odd positive number.")

    pad = int((window - 1) / 2)
    padded_data = np.pad(data, pad, mode='edge')
    result = np.zeros_like(data)

    for i in range(len(data)):
        result[i] = np.median(padded_data[i : i + window])
    return result

def position_medfilt(positions, window):
    """
    Apply a median filter to the x, y, size coordinates of detected regions.
    Args:
        positions (np.array): An array of shape (num_frames, 3) with [x, y, size].
        window (int): The median filter window size.
    Returns:
        np.array: The filtered positions.
    """
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Positions array must be of shape (num_frames, 3).")

    filtered_positions = np.copy(positions)

    # Filter only non-zero tracks to avoid distorting stationary regions
    for i in range(3): # x, y, size
        col = positions[:, i]
        # Create a boolean mask for non-zero values
        non_zero_mask = col != 0
        if np.any(non_zero_mask):
            # Apply median filter only to the non-zero elements
            filtered_values = medfilt(col[non_zero_mask], window)
            # Place the filtered values back into the original array
            filtered_positions[non_zero_mask, i] = filtered_values

    return filtered_positions
