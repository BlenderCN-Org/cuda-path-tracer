#include <nori/texture.h>
#include <filesystem/resolver.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

NORI_NAMESPACE_BEGIN

template <typename T>
class ImageTexture : public Texture<T> {
public:
    ImageTexture(const PropertyList &props);

    virtual std::string toString() const override;

    virtual T eval(const Point2f & uv) override {}

protected:
    float scale_x;
    Point2f m_delta;
    Vector2f m_scale;

    cv::Mat image;
};

template <>
float ImageTexture<float>::eval(const Point2f & uv)  {

    // rescale and shift
    float x = (uv.x()/m_scale.x()) - m_delta.x();
    float y = (uv.y()/m_scale.y()) - m_delta.y();

    x = x - floor(x);
    y = y - floor(y);

    int nCols = image.cols;

    int row = x*(image.rows);
    int col = y*(image.cols);

    unsigned char* p = image.data;

    unsigned char v = *(p + nCols*row + col+0);
    float val = v / 255.0f;

    return val;
}

template <>
Color3f ImageTexture<Color3f>::eval(const Point2f & uv)  {

    // rescale and shift
    float x = (uv.x()/m_scale.x()) - m_delta.x();
    float y = (uv.y()/m_scale.y()) - m_delta.y();

    x = x - floor(x);
    y = y - floor(y);

    int channels = image.channels();
	int nCols = image.cols * channels;

	//cout << x << " " << y << endl;

    int row = x*(image.rows);
    int col = y*(image.cols);

    unsigned char* p = image.data;

    unsigned char b = *(p + image.step*row + 3*col+0);
    unsigned char g = *(p + image.step*row + 3*col+1);
    unsigned char r = *(p + image.step*row + 3*col+2);

    float bf = b / 255.0f;
    float gf = g / 255.0f;
    float rf = r / 255.0f;

    return Color3f(rf,gf,bf);
}


template <>
ImageTexture<float>::ImageTexture(const PropertyList &props) {

    m_delta = props.getPoint2("delta", Point2f(0));
    m_scale = props.getVector2("scale", Vector2f(1));

    filesystem::path filename = getFileResolver()->resolve(props.getString("filename"));
    image = cv::imread(filename.str(), CV_LOAD_IMAGE_GRAYSCALE);
    if (!image.data)
    {
        throw NoriException("Image %s could not be found!", filename); \
    }
    /*cv::namedWindow("test");
    cv::imshow("img", image);
    cv::waitKey();*/
}
template <>
ImageTexture<Color3f>::ImageTexture(const PropertyList &props) {

    m_delta = props.getPoint2("delta", Point2f(0));
    m_scale = props.getVector2("scale", Vector2f(1));

    filesystem::path filename = getFileResolver()->resolve(props.getString("filename"));
    image = cv::imread(filename.str(), CV_LOAD_IMAGE_COLOR); // BGR format
    if (!image.data)
    {
        throw NoriException("Image %s could not be found!", filename); \
    }

    /*cv::namedWindow("test");
    cv::imshow("img", image);
    cv::waitKey();*/
}


template <>
std::string ImageTexture<float>::toString() const {
    return tfm::format(
            "ImageTexture[]");
            
}

template <>
std::string ImageTexture<Color3f>::toString() const {
    return tfm::format(
            "ImageTexture[]");
}

NORI_REGISTER_TEMPLATED_CLASS(ImageTexture, float, "image_float")
NORI_REGISTER_TEMPLATED_CLASS(ImageTexture, Color3f, "image_color")
NORI_NAMESPACE_END
