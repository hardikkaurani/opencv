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
from typing import Optional, Tuple


def detect_edges(
    image_path: str,
    lower_threshold: int = 50,
    upper_threshold: int = 150,
    display: bool = True,
    blur_kernel: int = 5,
    save_output: Optional[str] = None
) -> Optional[cv2.Mat]:
    """
    Detect edges in an image using Canny edge detection with preprocessing.

    Parameters
    ----------
    image_path : str
        Path to the image file to process
    lower_threshold : int, optional
        Lower threshold for Canny edge detection (default: 50)
    upper_threshold : int, optional
        Upper threshold for Canny edge detection (default: 150)
    display : bool, optional
        Whether to display the result in a window (default: True)
    blur_kernel : int, optional
        Kernel size for Gaussian blur preprocessing (default: 5, must be odd)
    save_output : Optional[str], optional
        Path to save the edge-detected image (default: None)

    Returns
    -------
    Optional[cv2.Mat]
        The edge-detected image, or None if image loading fails

    Raises
    ------
    FileNotFoundError
        If the image file does not exist
    ValueError
        If image loading fails or invalid parameters provided
    """
    # Validate image path exists
    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Validate threshold values
    if lower_threshold < 0 or upper_threshold < 0:
        raise ValueError("Thresholds must be non-negative values")
    if lower_threshold >= upper_threshold:
        raise ValueError(f"Lower threshold ({lower_threshold}) must be less than upper threshold ({upper_threshold})")

    # Validate blur kernel is odd
    if blur_kernel % 2 == 0:
        blur_kernel += 1

    # Load the image from file
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from: {image_path}")

    # Convert image to grayscale (removes color information)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray_image, (blur_kernel, blur_kernel), 1.0)

    # Apply Canny edge detection with specified thresholds
    edges = cv2.Canny(blurred, lower_threshold, upper_threshold)

    # Save output if requested
    if save_output:
        success = cv2.imwrite(save_output, edges)
        if success:
            print(f"✓ Edge map saved to: {save_output}")
        else:
            print(f"✗ Failed to save edge map to: {save_output}")

    # Display the results if requested
    if display:
        cv2.imshow('Original Image', image)
        cv2.imshow('Grayscale Image', gray_image)
        cv2.imshow('Detected Edges', edges)
        print("Press any key to close the windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return edges


def main() -> int:
    """
    Main entry point for the edge detection script.

    Accepts command-line arguments:
    - Image path (required or uses default 'img.jpg')
    - Lower threshold (optional, default: 50)
    - Upper threshold (optional, default: 150)

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Parse command-line arguments
        image_path = sys.argv[1] if len(sys.argv) > 1 else 'img.jpg'
        lower_threshold = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        upper_threshold = int(sys.argv[3]) if len(sys.argv) > 3 else 150

        # Create output filename based on input
        input_file = Path(image_path)
        output_path = input_file.stem + '_edges' + input_file.suffix

        print(f"Processing image: {image_path}")
        print(f"Thresholds: lower={lower_threshold}, upper={upper_threshold}")
        
        edges = detect_edges(
            image_path,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
            save_output=output_path
        )

        if edges is not None:
            print("✓ Edge detection completed successfully!")
            return 0
        else:
            print("✗ Edge detection failed!")
            return 1

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
