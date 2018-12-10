
#ifndef NORIGL_IMAGE_TEXTURE_H
#define NORIGL_IMAGE_TEXTURE_H

#include <nori/texture.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

NORI_NAMESPACE_BEGIN
template <typename T>
class ImageTexture : public Texture<T> {
public:
    __host__ ImageTexture(const PropertyList &props);
    __host__ virtual size_t getSize() const override {return sizeof(ImageTexture<T>);};
    __host__ virtual std::string toString() const override;
    __host__ virtual void gpuTransfer(NoriObject **) override;
    __host__ cv::Mat getImage();
    __device__ T eval(const Point2f & uv);

protected:
    float scale_x;
    Point2f m_delta;
    Vector2f m_scale;
    cv::Mat image; // image on cpu

    cudaTextureObject_t image_tex; // image on gpu
    unsigned char* gpu_data;
    size_t gpu_step;
};

NORI_NAMESPACE_END
#endif //NORIGL_IMAGE_TEXTURE_H
