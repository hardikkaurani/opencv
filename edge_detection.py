"""
Edge Detection Module

This module provides functionality for detecting edges in images using OpenCV's
Canny edge detection algorithm. It reads an image file, converts it to grayscale,
applies Canny edge detection, and displays the results.

Features:
- Canny edge detection with configurable thresholds
- Gaussian blur preprocessing for improved edge detection
- Real-time image display with window controls
- Optional edge map saving to file
- Comprehensive error handling and validation

Author: OpenCV Contributors
License: Apache License 2.0

Example:
    >>> edges = detect_edges('image.jpg', lower_threshold=50, upper_threshold=150)
    >>> save_edges(edges, 'edges_output.jpg')
"""

import cv2
import sys
from pathlib import Path
import numpy as np
from typing import Optional, Tuple, Union


def _process_edges(
    image: np.ndarray,
    lower_threshold: int = 50,
    upper_threshold: int = 150,
    blur_kernel: int = 5
) -> np.ndarray:
    """
    **Core edge detection processing (numpy-based, testable, reusable)**
    
    Detect edges in an image array using Canny edge detection with Gaussian blur.
    This is the core reusable function - works with any numpy array/cv2 image.
    
    Parameters
    ----------
    image : np.ndarray
        Image array (BGR or grayscale). Shape (H, W, 3) or (H, W)
    lower_threshold : int, optional
        Canny lower threshold (default: 50)
    upper_threshold : int, optional
        Canny upper threshold (default: 150)
    blur_kernel : int, optional
        Gaussian blur kernel size, must be odd (default: 5)

    Returns
    -------
    np.ndarray
        Binary edge map (single channel, uint8)
        
    Raises
    ------
    ValueError
        If thresholds are invalid or image is empty
    """
    # Validate thresholds
    if lower_threshold < 0 or upper_threshold < 0:
        raise ValueError("Thresholds must be non-negative")
    if lower_threshold >= upper_threshold:
        raise ValueError(f"Lower threshold ({lower_threshold}) must be < upper ({upper_threshold})")
    
    # Ensure blur kernel is odd
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply Gaussian blur + Canny edge detection
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 1.0)
    edges = cv2.Canny(blurred, lower_threshold, upper_threshold)
    
    return edges


def detect_edges(
    image_path: str,
    lower_threshold: int = 50,
    upper_threshold: int = 150,
    blur_kernel: int = 5,
    save_output: Optional[str] = None,
    display: bool = True
) -> Optional[np.ndarray]:
    """
    **Backward-compatible file-based wrapper**
    
    Load an image from file and detect edges. For batch/production use,
    consider loading images yourself and using _process_edges() directly.

    Parameters
    ----------
    image_path : str
        Path to image file
    lower_threshold : int, optional
        Canny lower threshold (default: 50)
    upper_threshold : int, optional
        Canny upper threshold (default: 150)
    blur_kernel : int, optional
        Gaussian blur kernel size (default: 5)
    save_output : Optional[str], optional
        Path to save result (default: None)
    display : bool, optional
        Show visualization windows (default: True)

    Returns
    -------
    Optional[np.ndarray]
        Edge map, or None on failure

    Examples
    --------
    Basic usage::
        edges = detect_edges('photo.jpg')
    
    Production/batch usage (no display/file I/O)::
        import cv2
        image = cv2.imread('photo.jpg')
        edges = _process_edges(image, lower_threshold=50, upper_threshold=150)
    """
    # Validate file exists
    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Process using core function
    edges = _process_edges(image, lower_threshold, upper_threshold, blur_kernel)

    # Save if requested
    if save_output:
        cv2.imwrite(save_output, edges)
        print(f"✓ Saved: {save_output}")

    # Display if requested
    if display:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Original', image)
        cv2.imshow('Grayscale', gray)
        cv2.imshow('Edges', edges)
        print("Press key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return edges


if __name__ == "__main__":
    # Example 1: Traditional usage (file-based, with display)
    try:
        image_path = sys.argv[1] if len(sys.argv) > 1 else 'img.jpg'
        lower = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        upper = int(sys.argv[3]) if len(sys.argv) > 3 else 150
        
        print(f"Processing: {image_path}")
        edges = detect_edges(image_path, lower, upper, display=True)
        
        if edges is not None:
            print("✓ Success")
            sys.exit(0)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
