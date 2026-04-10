"""
Edge Detection Module

This module provides functionality for detecting edges in images using OpenCV's
Canny edge detection algorithm. It reads an image file, converts it to grayscale,
applies Canny edge detection, and displays the results.

Author: OpenCV Contributors
License: Apache License 2.0
"""

import cv2
import sys
from pathlib import Path
from typing import Optional


def detect_edges(
    image_path: str,
    lower_threshold: int = 100,
    upper_threshold: int = 200,
    display: bool = True
) -> Optional[cv2.Mat]:
    """
    Detect edges in an image using Canny edge detection.

    Parameters
    ----------
    image_path : str
        Path to the image file to process
    lower_threshold : int, optional
        Lower threshold for Canny edge detection (default: 100)
    upper_threshold : int, optional
        Upper threshold for Canny edge detection (default: 200)
    display : bool, optional
        Whether to display the result in a window (default: True)

    Returns
    -------
    Optional[cv2.Mat]
        The edge-detected image, or None if image loading fails

    Raises
    ------
    FileNotFoundError
        If the image file does not exist
    """
    # Validate image path exists
    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Load the image from file
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from: {image_path}")

    # Convert image to grayscale (removes color information)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection with specified thresholds
    edges = cv2.Canny(gray_image, lower_threshold, upper_threshold)

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

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Use default image file or command-line argument
        image_path = sys.argv[1] if len(sys.argv) > 1 else 'img.jpg'

        print(f"Processing image: {image_path}")
        edges = detect_edges(image_path)

        if edges is not None:
            print("Edge detection completed successfully!")
            return 0
        else:
            print("Edge detection failed!")
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
