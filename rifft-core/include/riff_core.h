#ifndef RIFFT_CORE_H
#define RIFFT_CORE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RifftHandle* riff_handle_t;

typedef struct {
    unsigned height;
    unsigned width;
} riff_fft_shape_t;

riff_handle_t riff_create_handle(void);
void riff_free_handle(riff_handle_t handle);

int riff_fft2d_forward(riff_handle_t handle, void* data, unsigned height, unsigned width);
int riff_fft2d_inverse(riff_handle_t handle, void* data, unsigned height, unsigned width);
int riff_fft2d_fused_filter(
    riff_handle_t handle,
    void* data,
    const void* filter,
    unsigned height,
    unsigned width);

const char* riff_get_version(void);
const char* riff_get_backend_name(void);

#ifdef __cplusplus
}
#endif

#endif // RIFFT_CORE_H
