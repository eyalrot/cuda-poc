import numpy as np
import cv2

def process_image(image):
    """
    Process a 2D image with predefined 5x5 Gaussian filter and thresholding.
    Uses OpenCV's filter2D for convolution with zero padding.
    
    Args:
        image: 2D numpy array of type float32
    
    Returns:
        Processed image of same type (float32)
    """
    # Predefined 5x5 Gaussian filter coefficients (normalized)
    gaussian_kernel = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ], dtype=np.float32) / 256.0
    
    # Apply 5x5 Gaussian filter using OpenCV with zero padding
    # borderType=cv2.BORDER_CONSTANT with borderValue=0 provides zero padding
    filtered_image = cv2.filter2D(image, -1, gaussian_kernel, 
                                  borderType=cv2.BORDER_CONSTANT, 
                                  borderValue=0)
    
    # Apply threshold of 0.5
    thresholded_image = np.where(filtered_image > 0.5, filtered_image, 0.0)
    
    return thresholded_image.astype(np.float32)

if __name__ == "__main__":
    # Example usage
    # Create a sample 2D image
    sample_image = np.random.rand(100, 100).astype(np.float32)
    
    # Process the image
    result = process_image(sample_image)
    
    print(f"Input shape: {sample_image.shape}")
    print(f"Input dtype: {sample_image.dtype}")
    print(f"Output shape: {result.shape}")
    print(f"Output dtype: {result.dtype}")
    print(f"Input range: [{sample_image.min():.3f}, {sample_image.max():.3f}]")
    print(f"Output range: [{result.min():.3f}, {result.max():.3f}]")