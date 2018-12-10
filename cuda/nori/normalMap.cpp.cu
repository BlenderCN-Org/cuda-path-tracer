#include <nori/normalMap.h>
#include <filesystem/resolver.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

NORI_NAMESPACE_BEGIN


__device__ Vector3f NormalMap::eval(const Point2f& uv, const Frame& frame)  {

    // rescale and shift
    float x = (uv.x()/m_scale.x()) - m_delta.x();
    float y = (uv.y()/m_scale.y()) - m_delta.y();

    x = x - floor(x);
    y = y - floor(y);

    float4 rgb = tex2D<float4>(image_tex, y, x);

    Vector3f n((2.0f*rgb.x)-1.0f, (2.0f*rgb.y)-1.0f, rgb.z);
    n.normalize();
    return frame.toWorld(n);
}

#ifndef __CUDA_ARCH__

 __host__ NormalMap::NormalMap(const PropertyList &props) {

    modifierType = ENormalMap;

    m_delta = props.getPoint2("delta", Point2f(0));
    m_scale = props.getVector2("scale", Vector2f(1));

    filesystem::path filename = getFileResolver()->resolve(props.getString("filename"));
    image = cv::imread(filename.str(), CV_LOAD_IMAGE_COLOR); // BGR format

    if (!image.data)
    {
        throw NoriException("Image %s could not be found!", filename); \
    }

    cout << getSize() << endl;

}


 __host__ std::string NormalMap::toString() const {
    return tfm::format(
            "NormalMap[]");
}


__host__ void NormalMap::transferImage(unsigned char** target, cudaTextureObject_t* image_tex, cv::Mat& img)
{
    CHECK_ERROR( cudaMallocPitch(target, &gpu_step, img.elemSize() * img.cols, img.rows));
    CHECK_ERROR( cudaMemcpy2D(*target, gpu_step, img.data, img.step, img.cols * img.elemSize(), img.rows, cudaMemcpyHostToDevice));

    cudaChannelFormatDesc desc = cudaCreateChannelDesc(8,8,8,8, cudaChannelFormatKindUnsigned);

    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = *target;
    resDesc.res.pitch2D.pitchInBytes = gpu_step;
    resDesc.res.pitch2D.width = img.cols;
    resDesc.res.pitch2D.height = img.rows;
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


    CHECK_ERROR(cudaCreateTextureObject(image_tex, &resDesc, &texDesc, NULL));
}

 __host__ void NormalMap::gpuTransfer(NoriObject ** objects) {

    uchar* rgbaData = new uchar[image.total()*4];
    cv::Mat image_rgba(image.size(), CV_8UC4, rgbaData);
    cv::cvtColor(image, image_rgba, CV_BGR2RGBA, 4);

    /*cv::namedWindow("test");
    cv::imshow("image", image_rgba);
    cv::waitKey();*/

    transferImage(&gpu_data, &image_tex, image_rgba);
}
#endif

#ifndef __CUDA_ARCH__
    NORI_REGISTER_CLASS(NormalMap, "normal_map")
#endif
NORI_NAMESPACE_END
