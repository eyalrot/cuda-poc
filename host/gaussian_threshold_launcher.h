#ifndef GAUSSIAN_THRESHOLD_LAUNCHER_H
#define GAUSSIAN_THRESHOLD_LAUNCHER_H

namespace cuda_image {

/*
 * Launch Gaussian threshold kernel on device memory
 * 
 * @param d_input Device pointer to input image
 * @param d_output Device pointer to output image
 * @param width Image width
 * @param height Image height
 * @param use_shared_memory Whether to use shared memory optimization
 */
void launch_gaussian_threshold(
    const float* d_input,
    float* d_output,
    int width,
    int height,
    bool use_shared_memory = true
);

/*
 * Process image using CUDA (handles memory transfers)
 * 
 * @param h_input Host pointer to input image
 * @param h_output Host pointer to output image
 * @param width Image width
 * @param height Image height
 * @param use_shared_memory Whether to use shared memory optimization
 */
void process_image_cuda(
    const float* h_input,
    float* h_output,
    int width,
    int height,
    bool use_shared_memory = true
);

} // namespace cuda_image

#endif // GAUSSIAN_THRESHOLD_LAUNCHER_H