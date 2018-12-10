#include <nori/imagetexture.h>
#include <filesystem/resolver.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

NORI_NAMESPACE_BEGIN



template <>
__device__ Color3f ImageTexture<Color3f>::eval(const Point2f & uv)  {

    // rescale and shift
    float x = (uv.x()/m_scale.x()) - m_delta.x();
    float y = (uv.y()/m_scale.y()) - m_delta.y();

    x = x - floor(x);
    y = y - floor(y);

    float4 rgb = tex2D<float4>(image_tex, y, x);

    // from sRGB to linear RGB

    return Color3f(rgb.x, rgb.y, rgb.z).toLinearRGB();
}


template <>
 __host__ ImageTexture<Color3f>::ImageTexture(const PropertyList &props) {

    textureType = EImageTexture;

    m_delta = props.getPoint2("delta", Point2f(0));
    m_scale = props.getVector2("scale", Vector2f(1));

    filesystem::path filename = getFileResolver()->resolve(props.getString("filename"));
    
    image = cv::imread(filename.str(), CV_LOAD_IMAGE_COLOR); // BGR format

    //cv::cvtColor(image, image, CV_BGR2XYZ);

    /*cv::namedWindow("test");
    cv::imshow("img", image);
    cv::waitKey();*/

    if (!image.data)
    {
        throw NoriException("Image %s could not be found!", filename); \
    }
}

template <typename  T>
__host__ cv::Mat ImageTexture<T>::getImage(){
    return this->image;
}

template <>
 __host__ std::string ImageTexture<Color3f>::toString() const {
    return tfm::format(
            "ImageTexture[]");
}

template <>
 __host__ void ImageTexture<Color3f>::gpuTransfer(NoriObject ** objects) {

    // convert bgr to rgba (cuda supports only 4 dim image elements, no 3 dim elements :( )
    uchar* rgbaData = new uchar[image.total()*4];
    cv::Mat image_rgba(image.size(), CV_8UC4, rgbaData);
    cv::cvtColor(image, image_rgba, CV_BGR2RGBA, 4);

    /*cv::namedWindow("test");
    cv::imshow("img", image_rgba);
    cv::waitKey();*/

    CHECK_ERROR( cudaMallocPitch(&gpu_data, &gpu_step, image_rgba.elemSize() * image_rgba.cols, image_rgba.rows));
    CHECK_ERROR( cudaMemcpy2D(gpu_data, gpu_step, image_rgba.data, image_rgba.step, image_rgba.cols * image_rgba.elemSize(), image_rgba.rows, cudaMemcpyHostToDevice));

    cudaChannelFormatDesc desc = cudaCreateChannelDesc(8,8,8,8, cudaChannelFormatKindUnsigned);

    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = gpu_data;
    resDesc.res.pitch2D.pitchInBytes = gpu_step;
    resDesc.res.pitch2D.width = image_rgba.cols;
    resDesc.res.pitch2D.height = image_rgba.rows;
    resDesc.res.pitch2D.desc = desc;


    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]   = cudaAddressModeWrap;
    texDesc.addressMode[1]   = cudaAddressModeWrap;
    texDesc.filterMode       = cudaFilterModePoint;
    texDesc.filterMode       = cudaFilterModeLinear;
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.readMode         = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;


    CHECK_ERROR(cudaCreateTextureObject(&image_tex, &resDesc, &texDesc, NULL));

    //CHECK_ERROR(cudaBindTexture2D(0, &image_tex, gpu_data, &desc, image_rgba.cols, image_rgba.rows, gpu_step));
}

#ifndef __CUDA_ARCH__
    NORI_REGISTER_TEMPLATED_CLASS(ImageTexture, Color3f, "image_color")
#endif
NORI_NAMESPACE_END
