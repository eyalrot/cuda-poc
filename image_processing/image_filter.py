import numpy as np

def process_image(image):
    """
    Process a 2D image with predefined 5x5 Gaussian filter and thresholding.
    
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
    
    # Apply 5x5 Gaussian filter using convolution
    filtered_image = apply_convolution(image, gaussian_kernel)
    
    # Apply threshold of 0.5
    thresholded_image = np.where(filtered_image > 0.5, filtered_image, 0.0)
    
    return thresholded_image.astype(np.float32)

def apply_convolution(image, kernel):
    """
    Apply convolution with the given kernel.
    """
    kernel_size = kernel.shape[0]
    pad_size = kernel_size // 2
    
    # Pad the image with reflection
    padded_image = np.pad(image, pad_size, mode='reflect')
    
    # Apply convolution
    filtered = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            filtered[i, j] = np.sum(
                padded_image[i:i+kernel_size, j:j+kernel_size] * kernel
            )
    
    return filtered

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