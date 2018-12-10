#include <nori/bumpMap.h>
#include <filesystem/resolver.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

NORI_NAMESPACE_BEGIN


__device__ Vector3f BumpMap::eval(const Point2f& uv, const Frame& frame)  {

    // rescale and shift
    float x = (uv.x()/m_scale.x()) - m_delta.x();
    float y = (uv.y()/m_scale.y()) - m_delta.y();

    x = x - floor(x);
    y = y - floor(y);

    float Bu = tex2D<float>(image_tex_x, x, y);
    float Bv = tex2D<float>(image_tex_y, x, y);

    //printf("%.3f %.3f",Bu,Bv);

    Vector3f n = frame.n + Bu*frame.n.cross(frame.s) - Bv*frame.n.cross(frame.t);
    n.normalize();
    return n;
}

#ifndef __CUDA_ARCH__

 __host__ BumpMap::BumpMap(const PropertyList &props) {

    modifierType = EBumpMap;

    m_delta = props.getPoint2("delta", Point2f(0));
    m_scale = props.getVector2("scale", Vector2f(1));
    height = props.getFloat("height", 5);
    grad_delta = props.getFloat("grad_delta", 0);

    filesystem::path filename = getFileResolver()->resolve(props.getString("filename"));
    image = cv::imread(filename.str(), CV_LOAD_IMAGE_GRAYSCALE);

    if (!image.data)
    {
        throw NoriException("Image %s could not be found!", filename); \
    }

    cout << getSize() << endl;

}


 __host__ std::string BumpMap::toString() const {
    return tfm::format(
            "BumpMap[]");
}


__host__ void BumpMap::transferImage(unsigned char** target, cudaTextureObject_t* image_tex, cv::Mat& img)
{
    CHECK_ERROR( cudaMallocPitch(target, &gpu_step, img.elemSize() * img.cols, img.rows));
    CHECK_ERROR( cudaMemcpy2D(*target, gpu_step, img.data, img.step, img.cols * img.elemSize(), img.rows, cudaMemcpyHostToDevice));

    cudaChannelFormatDesc desc = cudaCreateChannelDesc(16,0,0,0, cudaChannelFormatKindSigned);

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

 __host__ void BumpMap::gpuTransfer(NoriObject ** objects) {

    cv::Mat grad_x, grad_y;

    // derivative parameters
    int ddepth = CV_16S;

    /// Gradient X
    //cv::Scharr( img, grad_x, ddepth, 1, 0, height, grad_delta, cv::BORDER_DEFAULT );
    cv::Sobel(image, grad_x, ddepth, 1, 0, 3, height, grad_delta, cv::BORDER_DEFAULT );

    /// Gradient Y
    //cv::Scharr( image, grad_y, ddepth, 0, 1, height, grad_delta, cv::BORDER_DEFAULT );
    cv::Sobel(image, grad_y, ddepth, 0, 1, 3, height, grad_delta, cv::BORDER_DEFAULT );

    /*cv::namedWindow("test");
    cv::imshow("image", grad_x);
    cv::waitKey();
    cv::imshow("image", grad_y);
    cv::waitKey();*/


    transferImage(&gpu_data_x, &image_tex_x, grad_x);
    transferImage(&gpu_data_y, &image_tex_y, grad_y);
}
#endif

#ifndef __CUDA_ARCH__
    NORI_REGISTER_CLASS(BumpMap, "bump_map")
#endif
NORI_NAMESPACE_END
