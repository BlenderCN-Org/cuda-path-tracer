

// workaround issue between gcc >= 4.7 and cuda 5.5
#if (defined __GNUC__) && (__GNUC__>4 || __GNUC_MINOR__>=7)
  #undef _GLIBCXX_ATOMIC_BUILTINS
  #undef _GLIBCXX_USE_INT128
#endif

#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX
#define EIGEN_TEST_FUNC cuda_basic
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

#include "main.h"
#include "cuda_common.h"

#include <Eigen/Eigenvalues>

// struct Foo{
//   EIGEN_DEVICE_FUNC
//   void operator()(int i, const float* mats, float* vecs) const {
//     using namespace Eigen;
//   //   Matrix3f M(data);
//   //   Vector3f x(data+9);
//   //   Map<Vector3f>(data+9) = M.inverse() * x;
//     Matrix3f M(mats+i/16);
//     Vector3f x(vecs+i*3);
//   //   using std::min;
//   //   using std::sqrt;
//     Map<Vector3f>(vecs+i*3) << x.minCoeff(), 1, 2;// / x.dot(x);//(M.inverse() *  x) / x.x();
//     //x = x*2 + x.y() * x + x * x.maxCoeff() - x / x.sum();
//   }
// };

template<typename T>
struct coeff_wise {
  EIGEN_DEVICE_FUNC
  void operator()(int i, const typename T::Scalar* in, typename T::Scalar* out) const
  {
    using namespace Eigen;
    T x1(in+i);
    T x2(in+i+1);
    T x3(in+i+2);
    Map<T> res(out+i*T::MaxSizeAtCompileTime);
    
    res.array() += (in[0] * x1 + x2).array() * x3.array();
  }
};

template<typename T>
struct redux {
  EIGEN_DEVICE_FUNC
  void operator()(int i, const typename T::Scalar* in, typename T::Scalar* out) const
  {
    using namespace Eigen;
    int N = 6;
    T x1(in+i);
    out[i*N+0] = x1.minCoeff();
    out[i*N+1] = x1.maxCoeff();
    out[i*N+2] = x1.sum();
    out[i*N+3] = x1.prod();
//     out[i*N+4] = x1.colwise().sum().maxCoeff();
//     out[i*N+5] = x1.rowwise().maxCoeff().sum();
  }
};

template<typename T1, typename T2>
struct prod {
  EIGEN_DEVICE_FUNC
  void operator()(int i, const typename T1::Scalar* in, typename T1::Scalar* out) const
  {
    using namespace Eigen;
    typedef Matrix<typename T1::Scalar, T1::RowsAtCompileTime, T2::ColsAtCompileTime> T3;
    T1 x1(in+i);
    T2 x2(in+i+1);
    Map<T3> res(out+i*T3::MaxSizeAtCompileTime);
    res += in[i] * x1 * x2;
  }
};

template<typename T1, typename T2>
struct diagonal {
  EIGEN_DEVICE_FUNC
  void operator()(int i, const typename T1::Scalar* in, typename T1::Scalar* out) const
  {
    using namespace Eigen;
    T1 x1(in+i);
    Map<T2> res(out+i*T2::MaxSizeAtCompileTime);
    res += x1.diagonal();
  }
};

template<typename T>
struct eigenvalues {
  EIGEN_DEVICE_FUNC
  void operator()(int i, const typename T::Scalar* in, typename T::Scalar* out) const
  {
    using namespace Eigen;
    typedef Matrix<typename T::Scalar, T::RowsAtCompileTime, 1> Vec;
    T M(in+i);
    Map<Vec> res(out+i*Vec::MaxSizeAtCompileTime);
    T A = M*M.adjoint();
    SelfAdjointEigenSolver<T> eig;
    eig.computeDirect(M);
    res = eig.eigenvalues();
  }
};

void test_cuda_basic()
{
  ei_test_init_cuda();
  
  int nthreads = 100;
  Eigen::VectorXf in, out;
  
  #ifndef __CUDA_ARCH__
  int data_size = nthreads * 16;
  in.setRandom(data_size);
  out.setRandom(data_size);
  #endif
  
  CALL_SUBTEST( run_and_compare_to_cuda(coeff_wise<Vector3f>(), nthreads, in, out) );
  CALL_SUBTEST( run_and_compare_to_cuda(coeff_wise<Array44f>(), nthreads, in, out) );
  
  CALL_SUBTEST( run_and_compare_to_cuda(redux<Array4f>(), nthreads, in, out) );
  CALL_SUBTEST( run_and_compare_to_cuda(redux<Matrix3f>(), nthreads, in, out) );
  
  CALL_SUBTEST( run_and_compare_to_cuda(prod<Matrix3f,Matrix3f>(), nthreads, in, out) );
  CALL_SUBTEST( run_and_compare_to_cuda(prod<Matrix4f,Vector4f>(), nthreads, in, out) );
  
  CALL_SUBTEST( run_and_compare_to_cuda(diagonal<Matrix3f,Vector3f>(), nthreads, in, out) );
  CALL_SUBTEST( run_and_compare_to_cuda(diagonal<Matrix4f,Vector4f>(), nthreads, in, out) );
  
  CALL_SUBTEST( run_and_compare_to_cuda(eigenvalues<Matrix3f>(), nthreads, in, out) );
  CALL_SUBTEST( run_and_compare_to_cuda(eigenvalues<Matrix2f>(), nthreads, in, out) );

}
/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Wenzel Jakob, Romain Prévost

    Nori is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Nori is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <nori/mesh.h>
#include <nori/bbox.h>
#include <nori/bsdf.h>
#include <nori/emitter.h>
#include <nori/warp.h>
#include <Eigen/Geometry>
#include <cuda_runtime_api.h>

NORI_NAMESPACE_BEGIN

Mesh::Mesh() : m_V(MatrixXf().data(),0,0),m_N(NULL,0,0),m_UV(NULL,0,0),m_F(NULL,0,0) {
    shapeType = ETMesh;
}

void Mesh::activate() {
    Shape::activate();

    m_pdf.reserve(getPrimitiveCount());
    for(uint32_t i = 0 ; i < getPrimitiveCount() ; ++i) {
        m_pdf.append(surfaceArea(i));
    }
    m_pdf.normalize();
    /**
     * make the class usable on cpu
     */
    //this->m_V =  m_V;
    //this->m_N = m_N;
    //this->m_UV = m_UV;
    //this->m_F = m_F;

}

__device__ void Mesh::sampleSurface(ShapeQueryRecord & sRec, const Point2f & sample) const {
    Point2f s = sample;

    size_t idT = m_pdf.sampleReuse(s.x());



    Vector3f bc = Warp::squareToUniformTriangle(s);

    sRec.p = getInterpolatedVertex(idT,bc);
    if (m_N.size() > 0) {
        sRec.n = getInterpolatedNormal(idT, bc);
    }
    else {
        Point3f p0 = m_V.col(m_F(0, idT));
        Point3f p1 = m_V.col(m_F(1, idT));
        Point3f p2 = m_V.col(m_F(2, idT));
        Normal3f n = (p1-p0).cross(p2-p0).normalized();
        sRec.n = n;
    }
    sRec.pdf = m_pdf.getNormalization();
}
__device__ float Mesh::pdfSurface(const ShapeQueryRecord & sRec) const {
    return m_pdf.getNormalization();
}

__device__ Point3f Mesh::getInterpolatedVertex(uint32_t index, const Vector3f &bc) const {
    return (bc.x() * m_V.col(m_F(0, index)) +
            bc.y() * m_V.col(m_F(1, index)) +
            bc.z() * m_V.col(m_F(2, index)));
}

__device__ Normal3f Mesh::getInterpolatedNormal(uint32_t index, const Vector3f &bc) const {
    return (bc.x() * m_N.col(m_F(0, index)) +
            bc.y() * m_N.col(m_F(1, index)) +
            bc.z() * m_N.col(m_F(2, index))).normalized();
}

__device__ float Mesh::surfaceArea(uint32_t index) const {
    uint32_t i0 = m_F(0, index), i1 = m_F(1, index), i2 = m_F(2, index);

    const Point3f p0 = m_V.col(i0), p1 = m_V.col(i1), p2 = m_V.col(i2);

    return 0.5f * Vector3f((p1 - p0).cross(p2 - p0)).norm();
}

__device__ bool Mesh::rayIntersect(uint32_t index, const Ray3f &ray, float &u, float &v, float &t) const {

    uint32_t i0 = m_F(0, index), i1 = m_F(1, index), i2 = m_F(2, index);
    const Point3f p0 = m_V.col(i0), p1 = m_V.col(i1), p2 = m_V.col(i2);

    /* Find vectors for two edges sharing v[0] */
    Vector3f edge1 = p1 - p0, edge2 = p2 - p0;

    /* Begin calculating determinant - also used to calculate U parameter */
    Vector3f pvec = ray.d.cross(edge2);

    /* If determinant is near zero, ray lies in plane of triangle */
    float det = edge1.dot(pvec);

    if (det > -1e-8f && det < 1e-8f)
        return false;
    float inv_det = 1.0f / det;

    /* Calculate distance from v[0] to ray origin */
    Vector3f tvec = ray.o - p0;

    /* Calculate U parameter and test bounds */
    u = tvec.dot(pvec) * inv_det;
    if (u < 0.0 || u > 1.0)
        return false;

    /* Prepare to test V parameter */
    Vector3f qvec = tvec.cross(edge1);

    /* Calculate V parameter and test bounds */
    v = ray.d.dot(qvec) * inv_det;
    if (v < 0.0 || u + v > 1.0)
        return false;

    /* Ray intersects triangle -> compute t */
    t = edge2.dot(qvec) * inv_det;

    return t >= ray.mint && t <= ray.maxt;
}

__device__ void Mesh::setHitInformation(uint32_t index, const Ray3f &ray, Intersection & its) const {
    /* Find the barycentric coordinates */
    Vector3f bary;
    bary << 1-its.uv.sum(), its.uv;

    /* Vertex indices of the triangle */
    uint32_t idx0 = m_F(0, index), idx1 = m_F(1, index), idx2 = m_F(2, index);

    Point3f p0 = m_V.col(idx0), p1 = m_V.col(idx1), p2 = m_V.col(idx2);

    /* Compute the intersection positon accurately
       using barycentric coordinates */
    its.p = bary.x() * p0 + bary.y() * p1 + bary.z() * p2;

    /* Compute proper texture coordinates if provided by the mesh */
    if (m_UV.size() > 0)
        its.uv = bary.x() * m_UV.col(idx0) +
                 bary.y() * m_UV.col(idx1) +
                 bary.z() * m_UV.col(idx2);

    /* Compute the geometry frame */
    its.geoFrame = Frame((p1-p0).cross(p2-p0).normalized());

    if (m_N.size() > 0) {
        /* Compute the shading frame. Note that for simplicity,
           the current implementation doesn't attempt to provide
           tangents that are continuous across the surface. That
           means that this code will need to be modified to be able
           use anisotropic BRDFs, which need tangent continuity */

        its.shFrame = Frame(
                (bary.x() * m_N.col(idx0) +
                 bary.y() * m_N.col(idx1) +
                 bary.z() * m_N.col(idx2)).normalized());
    } else {
        its.shFrame = its.geoFrame;
    }
}

 __host__ BoundingBox3f Mesh::getBoundingBox(uint32_t index) const {
    BoundingBox3f result(m_V.col(m_F(0, index)));
    result.expandBy(m_V.col(m_F(1, index)));
    result.expandBy(m_V.col(m_F(2, index)));
    return result;
}

 __host__ Point3f Mesh::getCentroid(uint32_t index) const {
    return (1.0f / 3.0f) *
        (m_V.col(m_F(0, index)) +
         m_V.col(m_F(1, index)) +
         m_V.col(m_F(2, index)));
}


std::string Mesh::toString() const {
    return tfm::format(
        "Mesh[\n"
        "  name = \"%s\",\n"
        "  vertexCount = %i,\n"
        "  triangleCount = %i,\n"
        "  bsdf = %s,\n"
        "  emitter = %s\n"
        "]",
        m_name,
        m_V.cols(),
        m_F.cols(),
        m_bsdf ? indent(m_bsdf->toString()) : std::string("null"),
        m_emitter ? indent(m_emitter->toString()) : std::string("null")
    );
}
void Mesh::gpuTransfer(NoriObject **objects) {
    Shape::gpuTransfer(objects);
    /*we assume that we do not wan't to accces most of the stuff again on the cpu so we copy it over to the gpu*/
    void *pV,*pN,*pUV,*pF;
    //step 1 allocate alle the storage
    cudaMalloc(&pV,sizeof(float)*m_V.size());
    cudaMalloc(&pN,sizeof(float)*m_N.size());
    cudaMalloc(&pUV,sizeof(float)*m_UV.size());
    cudaMalloc(&pF,sizeof(uint32_t)*m_F.size());
    //step 2 copy
    cudaMemcpy(pV,m_V.data(),sizeof(float)*m_V.size(),cudaMemcpyHostToDevice);
    cudaMemcpy(pN,m_N.data(),sizeof(float)*m_N.size(),cudaMemcpyHostToDevice);
    cudaMemcpy(pUV,m_UV.data(),sizeof(float)*m_UV.size(),cudaMemcpyHostToDevice);
    cudaMemcpy(pF,m_F.data(),sizeof(uint32_t)*m_F.size(),cudaMemcpyHostToDevice);
    //step 3 change data pointer
    /**
     * Object will be broken for cpu use afterwards as all pointer
     */
   new(&this->m_V) Eigen::Map<MatrixXf> ((float *) pV,m_V.rows(),m_V.cols());
   new(&this->m_N) Eigen::Map<MatrixXf> ((float *) pN,m_N.rows(),m_N.cols());
   new(&this->m_UV)Eigen::Map<MatrixXf> ((float *) pUV,m_UV.rows(),m_UV.cols());
   new(&this->m_F) Eigen::Map<MatrixXu> ((uint32_t *) pF,m_F.rows(),m_F.cols());

    //last part
    m_pdf.transferGpu();
};

NORI_NAMESPACE_END

#include <nori/imagetexture.h>
#include <filesystem/resolver.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

NORI_NAMESPACE_BEGIN


template <>
 __device__ float ImageTexture<float>::eval(const Point2f & uv)  {

    // rescale and shift
    float x = (uv.x()/m_scale.x()) - m_delta.x();
    float y = (uv.y()/m_scale.y()) - m_delta.y();

    x = x - floor(x);
    y = y - floor(x);

    /*int nCols = image.cols;

    int row = x*(image.rows);
    int col = y*(image.cols);

    unsigned char* p = image.data;

    unsigned char v = *(p + nCols*row + col+0);
    float val = v / 255.0f;*/

    //return val;
    return 1;
}

template <>
__device__ Color3f ImageTexture<Color3f>::eval(const Point2f & uv)  {

    // rescale and shift
    float x = (uv.x()/m_scale.x()) - m_delta.x();
    float y = (uv.y()/m_scale.y()) - m_delta.y();

    x = x - floor(x);
    y = y - floor(x);

    //uchar4 rgb = tex2D<uchar4>(image_tex, x, y);
        float4 rgb = tex2D<float4>(image_tex, x, y);

    /*int channels = image.channels();
    int nCols = image.cols * channels;

    int row = x*(image.rows);
    int col = y*(image.cols);

    unsigned char* p = image.data;

    unsigned char b = *(p + nCols*row + 3*col+0);
    unsigned char g = *(p + nCols*row + 3*col+1);
    unsigned char r = *(p + nCols*row + 3*col+2);*/

    float rf = rgb.x / 255.0f;
    float gf = rgb.y / 255.0f;
    float bf = rgb.z / 255.0f;

    return Color3f(rgb.x,rgb.y,rgb.z);
    return Color3f(rf,gf,bf);
}


template <>
__host__ ImageTexture<float>::ImageTexture(const PropertyList &props) {

    textureType = EImageTexture;

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
 __host__ ImageTexture<Color3f>::ImageTexture(const PropertyList &props) {

    textureType = EImageTexture;

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
__host__ std::string ImageTexture<float>::toString() const {
    return tfm::format(
            "ImageTexture[]");

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

template <>
__host__ void ImageTexture<float>::gpuTransfer(NoriObject ** objects) {
}

#ifndef __CUDA_ARCH__
    NORI_REGISTER_TEMPLATED_CLASS(ImageTexture, float, "image_float")
    NORI_REGISTER_TEMPLATED_CLASS(ImageTexture, Color3f, "image_color")
#endif
NORI_NAMESPACE_END
/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Romain Prévost

    Nori is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Nori is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <nori/shape.h>
#include <nori/bsdf.h>
#include <nori/emitter.h>
#include <nori/warp.h>

#include <Eigen/Geometry>

NORI_NAMESPACE_BEGIN


	__host__ Sphere::Sphere(const PropertyList & propList) {
		shapeType= ETSphere;
        m_position = propList.getPoint3("center", Point3f());
        m_radius = propList.getFloat("radius", 1.f);

        m_bbox.expandBy(m_position - Vector3f(m_radius));
        m_bbox.expandBy(m_position + Vector3f(m_radius));
    }

    __host__ BoundingBox3f Sphere::getBoundingBox(uint32_t index) const { return m_bbox; }

	__host__  Point3f Sphere::getCentroid(uint32_t index) const { return m_position; }

    __device__ bool Sphere::rayIntersect(uint32_t index, const Ray3f &ray, float &u, float &v, float &t) const {

		//float A = ray.d.transpose()*ray.d;
		// we assume A=1
		
		float B = 2*(ray.o - m_position).dot(ray.d);
		float C = -m_radius*m_radius + m_position.dot(m_position) + ray.o.dot(ray.o) - 2*m_position.dot(ray.o);

		//float discr = B*B - 4*A*C;
		float discr = B*B - 4*C;


		if (discr > 0)
		{
			// two solutions
			//float locT1 = (-B+sqrt(discr))/(2*A);
			//float locT2 = (-B-sqrt(discr))/(2*A);
			float locT1 = (-B+sqrt(discr))/(2.0);
			float locT2 = (-B-sqrt(discr))/(2.0);
			float locT = locT2;
			
			bool t1Valid = locT1 >= ray.mint && locT1 <= ray.maxt;
			bool t2Valid = locT2 >= ray.mint && locT2 <= ray.maxt;

			if (t1Valid && t2Valid)
			{
				locT = stdmin(locT1, locT2);
			}
			else if (t1Valid)
			{
				locT = locT1;
			}
			else if (t2Valid)
			{
				locT = locT2;
			}
			else
			{
				return false;
			}

			//if (locT2 >= 0) locT = locT2;
			//if (locT2 >= 0) locT = stdmin(locT, locT2);
			//else std::cout << "err" << endl;
			
			//if (locT >= ray.mint && locT <= ray.maxt)
			{
				t = locT;
				return true;
			}
		}
		else if (discr == 0)
		{
			// single solution
			float locT = -B/(2.0);

			if (locT >= ray.mint && locT < ray.maxt)
			{
				t = locT;
				return true;
			}
		}
		else
		{
			// no solution
			return false;
		}


        return false;

    }

	__device__  void Sphere::setHitInformation(uint32_t index, const Ray3f &ray, Intersection & its) const  {

		its.p = ray(its.t); // eval ray at time t
		its.shFrame = Frame((its.p - m_position).normalized());

		if ((ray(ray.mint) - m_position).squaredNorm() <= m_radius*m_radius)
		{
			//its.shFrame = Frame((-its.p + m_position).normalized());
		}

		its.geoFrame = its.shFrame;

		Point3f p_loc = its.p - m_position;

        
        // fix slightly out of boundary values
        float frac = p_loc.z()/m_radius;
        if (frac < -1)
            frac = -1;
        else if (frac > 1)
            frac = 1;

		float theta = acosf(frac) / (M_PI);
		float phi = (atan2f(p_loc.y(),p_loc.x()) + M_PI) / (2*M_PI);

		its.uv = Point2f(phi, theta);
    }

	__device__ void Sphere::sampleSurface(ShapeQueryRecord & sRec, const Point2f & sample) const  {
        Vector3f q = Warp::squareToUniformSphere(sample);
        sRec.p = m_position + m_radius * q;
        sRec.n = q;
        //sRec.pdf = std::pow(1.f/m_radius,2) * Warp::squareToUniformSpherePdf(Vector3f(0.0f,0.0f,1.0f));
        sRec.pdf = 1.0/(4.0*M_PI*m_radius*m_radius); //* Warp::squareToUniformSpherePdf(Vector3f(0.0f,0.0f,1.0f));
    }
	__device__ float Sphere::pdfSurface(const ShapeQueryRecord & sRec)  const {
        return powf(1.f/m_radius,2) * Warp::squareToUniformSpherePdf(Vector3f(0.0f,0.0f,1.0f));
    }


    __host__  std::string Sphere::toString() const  {
        return tfm::format(
                "Sphere[\n"
                "  center = %s,\n"
                "  radius = %f,\n"
                "  bsdf = %s,\n"
                "  emitter = %s\n"
                "]",
                m_position.toString(),
                m_radius,
                m_bsdf ? indent(m_bsdf->toString()) : std::string("null"),
                m_emitter ? indent(m_emitter->toString()) : std::string("null"));
    }


#ifndef __CUDA_ARCH__
	NORI_REGISTER_CLASS(Sphere, "sphere");
#endif
NORI_NAMESPACE_END
/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Wenzel Jakob, Romain Prévost

    Nori is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Nori is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/


#include <nori/diffuse.h>
#include <cuda_runtime_api.h>
#include <nori/texture.h>
#include <nori/consttexture.h>
#include <nori/imagetexture.h>
#include <nori/independent.h>
NORI_NAMESPACE_BEGIN

/**
 * \brief Diffuse / Lambertian BRDF model
 */

#ifndef __CUDA_ARCH__
    __host__ Diffuse::Diffuse(const PropertyList &propList) : m_albedo(nullptr) {
        this->isDiffuse = true;
        this->bsdfType = TDiffuse;
        if(propList.has("albedo")) {
            PropertyList l;
            l.setColor("value", propList.getColor("albedo"));
            m_albedo = static_cast<
                    Texture<
                            Color3f> *>(
                    NoriObjectFactory::createInstance("constant_color", l));
        }
    }
   __host__  Diffuse::~Diffuse() {
        //we assume that transfer has been called before, say cuda that albedo is not needed any longer
        cudaFree(m_albedo);

        //delete m_albedo;
    }


	/// Add texture for the albedo
    __host__ void Diffuse::addChild(NoriObject *obj)  {
        switch (obj->getClassType()) {
            case ETexture:
                if(obj->getIdName() == "albedo") {
                    if (m_albedo)
                        throw NoriException("There is already an albedo defined!");
                    m_albedo = static_cast<Texture<Color3f> *>(obj);
                }
                else {
                    throw NoriException("The name of this texture does not match any field!");
                }
                break;

            default:
                throw NoriException("Diffuse::addChild(<%s>) is not supported!",
                                    classTypeName(obj->getClassType()));
        }
    }
    __host__ void Diffuse::activate()  {
        if(!m_albedo) {
            PropertyList l;
            l.setColor("value", Color3f(0.5f));
            m_albedo = static_cast<Texture<Color3f> *>(NoriObjectFactory::createInstance("constant_color", l));
            m_albedo->activate();
        }
    }


    __host__ void Diffuse::gpuTransfer(NoriObject ** objects) {
        this->m_albedo = (Texture <Color3f> *) objects[m_albedo->objectId];
        //do not delete object now
        //delete tmp;
    }

#endif

    /// Evaluate the BRDF model
    __device__ Color3f Diffuse::eval(const BSDFQueryRecord &bRec) const  {
        /* This is a smooth BRDF -- return zero if the measure
           is wrong, or when queried for illumination on the backside */
        if (bRec.measure != ESolidAngle
            || Frame::cosTheta(bRec.wi) <= 0
            || Frame::cosTheta(bRec.wo) <= 0)
			return 0.0f;


        /* The BRDF is simply the albedo / pi */
        return CallTexture(m_albedo,Color3f,eval,bRec.uv) * INV_PI;
    }

    /// Compute the density of \ref sample() wrt. solid angles
     __device__ float Diffuse::pdf(const BSDFQueryRecord &bRec) const  {
        /* This is a smooth BRDF -- return zero if the measure
           is wrong, or when queried for illumination on the backside */
        if (bRec.measure != ESolidAngle
            || Frame::cosTheta(bRec.wi) <= 0
            || Frame::cosTheta(bRec.wo) <= 0)
			return 0.0f;



        /* Importance sampling density wrt. solid angles:
           cos(theta) / pi.

           Note that the directions in 'bRec' are in local coordinates,
           so Frame::cosTheta() actually just returns the 'z' component.
        */
        float r = INV_PI * Frame::cosTheta(bRec.wo);
        //printf("%f\n",r);
        return r;
    }

    /// Draw a a sample from the BRDF model
     __device__ Color3f Diffuse::sample(BSDFQueryRecord &bRec, const Point2f &sample) const  {

        if (Frame::cosTheta(bRec.wi) <= 0)
            return Color3f(0.0f);

        bRec.measure = ESolidAngle;

        /* Warp a uniformly distributed sample on [0,1]^2
           to a direction on a cosine-weighted hemisphere */
        bRec.wo = Warp::squareToCosineHemisphere(sample);

        /* Relative index of refraction: no change */
        bRec.eta = 1.0f;
        /* eval() / pdf() * cos(theta) = albedo. There
           is no need to call these functions. */
        return CallTexture(m_albedo,Color3f,eval,bRec.uv);
    }


    /// Return a human-readable summary
   __host__ std::string Diffuse::toString() const  {
        return tfm::format(
            "Diffuse[\n"
            "  albedo = %s\n"
            "]",
            m_albedo ? indent(m_albedo->toString()) : std::string("null")
        );
    }


#ifndef __CUDA_ARCH__
    NORI_REGISTER_CLASS(Diffuse, "diffuse");
#endif
NORI_NAMESPACE_END
/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Wenzel Jakob, Romain Prévost

    Nori is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Nori is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <nori/bsdf.h>
#include <nori/frame.h>
#include <nori/common.h>
#include <nori/warp.h>
#include <nori/texture.h>
#include <nori/consttexture.h>
#include <nori/imagetexture.h>
#include <nori/independent.h>
#include <nori/subsurface.h>

NORI_NAMESPACE_BEGIN

/**
 * \brief Subsurface / Lambertian BRDF model
 */

#ifndef __CUDA_ARCH__

__host__ Subsurface::Subsurface(const PropertyList &propList) : m_albedo(nullptr) {
    if(propList.has("albedo")) {
        PropertyList l;
        l.setColor("value", propList.getColor("albedo"));
        m_albedo = static_cast<Texture<Color3f> *>(NoriObjectFactory::createInstance("constant_color", l));
    }

    m_intIOR = propList.getFloat("intIOR", 1.3046f);
    m_extIOR = propList.getFloat("extIOR", 1.000277f);

    dmfp = propList.getColor("dmfp", 0.1f);

    isDiffuse = true;
    this->bsdfType = TSubsurface;
}

__host__ Subsurface::~Subsurface() {
    cudaFree(m_albedo);
}


/// Add texture for the albedo
__host__ void Subsurface::addChild(NoriObject *obj) {
    switch (obj->getClassType()) {
        case ETexture:
            if(obj->getIdName() == "albedo") {
                if (m_albedo)
                    throw NoriException("There is already an albedo defined!");
                m_albedo = static_cast<Texture<Color3f> *>(obj);
            }
            else {
                throw NoriException("The name of this texture does not match any field!");
            }
            break;

        default:
            throw NoriException("Subsurface::addChild(<%s>) is not supported!",
                                classTypeName(obj->getClassType()));
    }
}

__host__ void Subsurface::activate() {
    if(!m_albedo) {
        PropertyList l;
        l.setColor("value", Color3f(0.5f));
        m_albedo = static_cast<Texture<Color3f> *>(NoriObjectFactory::createInstance("constant_color", l));
        m_albedo->activate();
    }

    // approx (https://graphics.pixar.com/library/ApproxBSSRDF/paper.pdf)
    //s = 3.5f + 100.0f*pow((albedo - 0.33f), 4); // equation 8
    //s = 1.9f - albedo + 3.5f*pow((albedo - 0.8f), 2); // equation 6
    //s = 1.85f - albedo + 7.0f*abs(pow((albedo - 0.8f), 3)); // equation 5
    s = 1.0f; //(set s=1 is equal to: http://blog.selfshadow.com/publications/s2015-shading-course/burley/s2015_pbs_disney_bsdf_notes.pdf)
}

/// Return a human-readable summary
__host__ std::string Subsurface::toString() const {
    return tfm::format(
        "Subsurface[\n"
        "  albedo = %s\n"
        "]",
        m_albedo ? indent(m_albedo->toString()) : std::string("null")
    );
}

__host__ void Subsurface::gpuTransfer(NoriObject ** objects) {
    this->m_albedo = (Texture <Color3f> *) objects[m_albedo->objectId];
    //do not delete object now
    //delete tmp;
}

#endif

__device__ float Subsurface::colorNorm(const Color3f& c) const
{
    return sqrt(c.x()*c.x() + c.y()*c.y() + c.z()*c.z());
}

/// Evaluate the BRDF model
__device__ Color3f Subsurface::eval(const BSDFQueryRecord &bRec) const {

    //we want: return diffuseScatteringApprox() / bRec.dxPdf(); -> R(r) cancels and this gives:
    float F = fresnelTransmission(bRec);
    return F*CallTexture(m_albedo, Color3f, eval, bRec.uv)*INV_PI;
}

/*__device__ Vector3f Subsurface::expVec(const Vector3f &v) const {

    return Vector3f(exp(v.x()), exp(v.y()), exp(v.z()));
}*/

__device__ float Subsurface::diffuseScatterApproxCDF(float r) const
{
    float rsd = -r*colorNorm(s)/colorNorm(dmfp);
    return 1.0f - 0.25f*std::exp(rsd)-0.75f*std::exp(rsd/3.0f);
}

__device__ float Subsurface::diffuseScatterApproxCDFDerivative(float r) const
{
    float sd = colorNorm(s)/colorNorm(dmfp);
    float rsd = -r*sd;
    return sd * 0.25f* (std::exp(rsd) + std::exp(rsd/3.0f));
}

__device__ Point2f Subsurface::diffuseScatterPDFsample(const Point2f& sample2) const
{
    float sample = sample2.x();

    // newton find inverse of CDF
    float sdInv = colorNorm(dmfp)/colorNorm(s);
    float r0 = -log((1.0f - sample) * 4.0f/3.0f)*3.0f*sdInv;
    float r1 = -log((1.0f - sample) * 4.0f)*3.0f*sdInv;
    
    float err0 = fabs(diffuseScatterApproxCDF(r0) - sample);;
    float err1 = fabs(diffuseScatterApproxCDF(r1) - sample);

    if (err0 < err1)  r1=r0;
    else r1 = r0;

    //(cout << "|phi - cdf(r)|_0a=" << fabs(diffuseScatterApproxCDF(r0) - sample) << endl;
    //cout << "|phi - cdf(r)|_0b=" << fabs(diffuseScatterApproxCDF(r1) - sample) << endl;

    for (int i=0; i<2; ++i)
    {
        float f = (diffuseScatterApproxCDF(r0) - sample);
        float df = diffuseScatterApproxCDFDerivative(r0);
        r1 = r0 - f/df;
        r0 = r1;
    }

    //cout << "|phi - cdf(r)|_1=" << fabs(diffuseScatterApproxCDF(r0) - sample) << endl;
    
    float theta = 2.0f*M_PI*sample2.y();
    return Point2f(r0*cos(theta), r0*sin(theta));
}

__device__ float Subsurface::diffuseScatterPDF(const Point2f& p) const
{
    float sn = colorNorm(s);
    float dmfpn = colorNorm(dmfp);
    float r = p.norm();

    return sn*(std::exp(-sn*r/dmfpn) + std::exp(-sn*r/(3.0f*dmfpn)))/(8.0f*M_PI*dmfpn*r);
    //return diffuseScatterApproxCDFDerivative(r) / (2*M_PI*r);
}

__device__ Color3f Subsurface::diffuseScatteringApprox(const BSDFQueryRecord &bRec) const
{
    // approx (https://graphics.pixar.com/library/ApproxBSSRDF/paper.pdf equation 3)
    if (bRec.measure != ESolidAngle
        || Frame::cosTheta(bRec.wi) <= 0
        || Frame::cosTheta(bRec.wo) <= 0)
        return 0.0f;

    float r = bRec.probeDistance;

    Color3f albedo = CallTexture(m_albedo, Color3f, eval, bRec.uv);
    Color3f Rr = albedo*s*((-s*r/dmfp).exp() + (-s*r/(3.0f*dmfp)).exp())/(8.0f*M_PI*dmfp*r);
    float F = fresnelTransmission(bRec);
    return F*Rr*INV_PI;
}

__device__ float Subsurface::fresnelTransmission(const BSDFQueryRecord &bRec) const
{
    float Ft_i = 1.0f-fresnel(Frame::cosTheta(bRec.wi), m_extIOR, m_intIOR);
    float Ft_o = 1.0f-fresnel(Frame::cosTheta(bRec.wo), m_extIOR, m_intIOR);
    return Ft_i*Ft_o;
}

/// Compute the density of \ref sample() wrt. solid angles
__device__ float Subsurface::pdf(const BSDFQueryRecord &bRec) const {
    /* This is a smooth BRDF -- return zero if the measure
       is wrong, or when queried for illumination on the backside */
    if (bRec.measure != ESolidAngle
        || Frame::cosTheta(bRec.wi) <= 0
        || Frame::cosTheta(bRec.wo) <= 0)
        return 0.0f;


    /* Importance sampling density wrt. solid angles:
       cos(theta) / pi.

       Note that the directions in 'bRec' are in local coordinates,
       so Frame::cosTheta() actually just returns the 'z' component.
    */

    return INV_PI * Frame::cosTheta(bRec.wo); //* bRec.dxAxisPdf;
}

__device__ void Subsurface::sampleDx(BSDFQueryRecord &bRec, const Intersection& its, Sampler& sampler) const
{
    // scatter ray test
    Ray3f traceRay;// = bRec.probeRay;

    float thres = CallSampler(sampler, next1D);
    float k = 0.5f/4.0f;

    if (thres < 0.5f)
        traceRay.d = -its.shFrame.n; // to surface parallel to normal
    else if (thres < 0.5f+k)
        traceRay.d = its.shFrame.t; 
    else if (thres < 0.5f+2*k)
        traceRay.d = -its.shFrame.t; 
    else if (thres < 0.5f+3*k)
        traceRay.d = its.shFrame.s; 
    else 
        traceRay.d = -its.shFrame.s; 
    
    //bRec.dxAxisPdf = 1.0f / 5.0f;

    Point2f dx  = diffuseScatterPDFsample(CallSampler(sampler,next2D));
    //cout << dx.toString() << endl;
    Vector3f disk3d(dx.x(), dx.y(), 0);
    Frame frame(-traceRay.d);
    traceRay.o = its.p + frame.toWorld(disk3d);
    
    float eps = 1e-4;
    traceRay.mint = -eps;
    traceRay.maxt = dx.norm()+eps;
    traceRay.update();
    //cout << bRec.dx.toString() << endl;

    bRec.probeRay = traceRay;
    bRec.probe = true;
}


/// Draw a a sample from the BRDF model
__device__ Color3f Subsurface::sample(BSDFQueryRecord &bRec, const Point2f &sample) const
{
    if (Frame::cosTheta(bRec.wi) <= 0)
        return 0.0f;

    float costhetai = Frame::cosTheta(bRec.wi);

    float eta = m_extIOR / m_intIOR; 
    bRec.eta = eta;

    // diffuse scattering
    bRec.isDelta = false;
    bRec.wo = Warp::squareToCosineHemisphere(sample);

    float F = fresnelTransmission(bRec);

    // we want:
    // INV_PI*albedo*R(r)*costhetaoi/ (R(r)*INV_PI*costhetaoi*bRec.dxAxisPdf)
    // but almost all terms cancels except:
    return F*CallTexture(m_albedo, Color3f, eval, bRec.uv); // bRec.dxAxisPdf;
}


#ifndef __CUDA_ARCH__
    NORI_REGISTER_CLASS(Subsurface, "subsurface");
#endif
NORI_NAMESPACE_END
/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Wenzel Jakob

    Nori is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Nori is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <nori/warp.h>
#include <nori/independent.h>
//class Independent;
NORI_NAMESPACE_BEGIN

// CYLINDER of height 1
__device__ __host__ Vector3f Warp::squareToUniformCylinder(const Point2f& sample)
{
	float phi = 2*M_PI*sample.x();
	float z = sample.y();
	return Vector3f(cos(phi), sin(phi), z);
}


__device__ Vector3f Warp::sampleUniformHemisphere(Sampler sampler, const Normal3f &pole) {
    // Naive implementation using rejection sampling
    Vector3f v;
    do {
        v.x() = 1.f - 2.f * CallSampler(sampler,next1D);
        v.y() = 1.f - 2.f * CallSampler(sampler,next1D);
        v.z() = 1.f - 2.f * CallSampler(sampler,next1D);
    } while (v.squaredNorm() > 1.f);

    if (v.dot(pole) < 0.f)
        v = -v;
    v /= v.norm();

    return v;
}

// SQUARE
__device__ __host__  Point2f Warp::squareToUniformSquare(const Point2f &sample) {
    return sample;
}

__device__ __host__  float Warp::squareToUniformSquarePdf(const Point2f &sample) {
    return (sample.minCoeff() >= 0 && sample.maxCoeff() <= 1) ? 1.0f : 0.0f;
}

// DISK
__device__ __host__ Point2f Warp::squareToUniformDisk(const Point2f &sample) {
    float r = sqrtf(sample.x());
	float theta = 2*M_PI*sample.y();
	return Point2f(r*cos(theta), r*sin(theta));
}

__device__ __host__ float Warp::squareToUniformDiskPdf(const Point2f &p) {
	float r = p.norm();
	if (r<=1.0f)
	{
		return 1.0f/M_PI;
	}
	else return 0;
}

// CAP
__device__ __host__ Vector3f Warp::squareToUniformSphereCap(const Point2f &sample, float cosThetaMax) {
	Vector3f cyl = Warp::squareToUniformCylinder(sample);
	float offset = cosThetaMax;
	float height = 1.0f - offset;

	float z = offset + cyl.z()*height;
	float r = sqrtf(1-(z*z));
	return Vector3f(cyl.x()*r, cyl.y()*r, z);
}

__device__ __host__ float Warp::squareToUniformSphereCapPdf(const Vector3f &v, float cosThetaMax) {

	//float z = offset + cyl.z()*height;

	if (v.z() < cosThetaMax)
	{
		return 0;
	}
	else
	{
		float h = 1.0f - cosThetaMax;
		float area = 2.0f*M_PI*h;
		return 1.0f/area; // evenly distributed over area of the cap
	}
}

// SPHERE
__device__ __host__ Vector3f Warp::squareToUniformSphere(const Point2f &sample) {
	Vector3f cyl = Warp::squareToUniformCylinder(sample);
	float z = 2.0f*cyl.z()-1.0f;
	float r = sqrtf(1.0f-(z*z));
	return Vector3f(cyl.x()*r, cyl.y()*r, z);
}

__device__ __host__ float Warp::squareToUniformSpherePdf(const Vector3f &v) {
	return 1.0f/(4.0f*M_PI); // evenly distributed over area of a sphere
}


// HEMISPHERE
__device__ __host__ Vector3f Warp::squareToUniformHemisphere(const Point2f &sample) {
	return Warp::squareToUniformSphereCap(sample, 0);
}

__device__ __host__ float Warp::squareToUniformHemispherePdf(const Vector3f &v) {
	return squareToUniformSphereCapPdf(v, 0);
}


// COSINE HEMISPHERE
__device__ __host__ Vector3f Warp::squareToCosineHemisphere(const Point2f &sample) {
	Point2f d = Warp::squareToUniformDisk(sample);
	return Vector3f(d.x(), d.y(), sqrtf(1.0f - d.x()*d.x() - d.y()*d.y()));
}

__device__ __host__ float Warp::squareToCosineHemispherePdf(const Vector3f &v) {

	if (v.z() >= 0)
		return v.z()/M_PI;
	else 
		return 0;
	
}


// BECKMANN
__device__ __host__ Vector3f Warp::squareToBeckmann(const Point2f &sample, float alpha) {

    float theta = acos(sqrtf(1.0f / ( 1.0f - alpha*alpha*log( 1.0f - sample.x()))));
	float phi = sample.y()*2.0f*M_PI;

	return Vector3f(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta));

}

__device__ __host__ float Warp::squareToBeckmannPdf(const Vector3f &m, float alpha) {

	//return 0;

	if (m.z() >= 0 && fabs(m.norm()-1) < float(1e-6))
	{
		float theta = acos(m.z());
		float phi = atan2(m.y(), m.x());

		float alpha2 = alpha*alpha;
		return exp(-powf(tan(theta), 2) / alpha2) / (M_PI*alpha2*powf(cos(theta), 3));
	}
	else return 0;

}


__device__ Vector3f Warp::squareToUniformTriangle(const Point2f &sample) {
	float su1 = sqrtf(sample.x());
    float u = 1.f - su1, v = sample.y() * su1;
    return Vector3f(u,v,1.f-u-v);
}

NORI_NAMESPACE_END
/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Wenzel Jakob, Romain Prévost

    Nori is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Nori is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <nori/bvh.h>
#include <nori/timer.h>
#include <tbb/tbb.h>
#include <Eigen/Geometry>
#include <atomic>

/*
 * =======================================================================
 *   WARNING    WARNING    WARNING    WARNING    WARNING    WARNING
 * =======================================================================
 *   Remember to put on SAFETY GOGGLES before looking at this file. You
 *   are most certainly not expected to read or understand any of it.
 * =======================================================================
 */

NORI_NAMESPACE_BEGIN

//Building the bvh on the CPU for the moment
#ifndef __CUDA_ARCH__
/* Bin data structure for counting triangles and computing their bounding box */
    struct Bins {
        static const int BIN_COUNT = 16;
        Bins() { memset(counts, 0, sizeof(uint32_t) * BIN_COUNT); }
        uint32_t counts[BIN_COUNT];
        BoundingBox3f bbox[BIN_COUNT];
    };

/**
 * \brief Build task for parallel BVH construction
 *
 * This class uses the task scheduling system of Intel' Thread Building Blocks
 * to parallelize the divide and conquer BVH build at all levels.
 *
 * The used methodology is roughly that described in
 * "Fast and Parallel Construction of SAH-based Bounding Volume Hierarchies"
 * by Ingo Wald (Proc. IEEE/EG Symposium on Interactive Ray Tracing, 2007)
 */
class BVHBuildTask : public tbb::task {
    private:
        BVH &bvh;
        uint32_t node_idx;
        uint32_t *start, *end, *temp;

    public:
        /// Build-related parameters
        enum {
            /// Switch to a serial build when less than 32 triangles are left
                    SERIAL_THRESHOLD = 32,

            /// Process triangles in batches of 1K for the purpose of parallelization
                    GRAIN_SIZE = 1000,

            /// Heuristic cost value for traversal operations
                    TRAVERSAL_COST = 1,

            /// Heuristic cost value for intersection operations
                    INTERSECTION_COST = 1
        };

    public:
        /**
         * Create a new build task
         *
         * \param bvh
         *    Reference to the underlying BVH
         *
         * \param node_idx
         *    Index of the BVH node that should be built
         *
         * \param start
         *    Start pointer into a list of triangle indices to be processed
         *
         * \param end
         *    End pointer into a list of triangle indices to be processed
         *
         *  \param temp
         *    Pointer into a temporary memory region that can be used for
         *    construction purposes. The usable length is <tt>end-start</tt>
         *    unsigned integers.
         */
        BVHBuildTask(BVH &bvh, uint32_t node_idx, uint32_t *start, uint32_t *end, uint32_t *temp)
                : bvh(bvh), node_idx(node_idx), start(start), end(end), temp(temp) { }

        task *execute() {
            uint32_t size = (uint32_t) (end-start);
            BVH::BVHNode &node = bvh.m_nodes[node_idx];

            /* Switch to a serial build when less than SERIAL_THRESHOLD triangles are left */
            if (size < SERIAL_THRESHOLD) {
                execute_serially(bvh, node_idx, start, end, temp);
                return nullptr;
            }

            /* Always split along the largest axis */
            int axis = node.bbox.getLargestAxis();
            float min = node.bbox.min[axis], max = node.bbox.max[axis],
                    inv_bin_size = Bins::BIN_COUNT / (max-min);

            /* Accumulate all triangles into bins */
            Bins bins = tbb::parallel_reduce(
                    tbb::blocked_range<uint32_t>(0u, size, GRAIN_SIZE),
                    Bins(),
                    /* MAP: Bin a number of triangles and return the resulting 'Bins' data structure */
                    [&](const tbb::blocked_range<uint32_t> &range, Bins result) {
                        for (uint32_t i = range.begin(); i != range.end(); ++i) {
                            uint32_t f = start[i];
                            float centroid = bvh.getCentroid(f)[axis];

                            int index = std::min(std::max(
                                    (int) ((centroid - min) * inv_bin_size), 0),
                                                 (Bins::BIN_COUNT - 1));

                            result.counts[index]++;
                            result.bbox[index].expandBy(bvh.getBoundingBox(f));
                        }
                        return result;
                    },
                    /* REDUCE: Combine two 'Bins' data structures */
                    [](const Bins &b1, const Bins &b2) {
                        Bins result;
                        for (int i=0; i < Bins::BIN_COUNT; ++i) {
                            result.counts[i] = b1.counts[i] + b2.counts[i];
                            result.bbox[i] = BoundingBox3f::merge(b1.bbox[i], b2.bbox[i]);
                        }
                        return result;
                    }
            );

            /* Choose the best split plane based on the binned data */
            BoundingBox3f bbox_left[Bins::BIN_COUNT];
            bbox_left[0] = bins.bbox[0];
            for (int i=1; i<Bins::BIN_COUNT; ++i) {
                bins.counts[i] += bins.counts[i-1];
                bbox_left[i] = BoundingBox3f::merge(bbox_left[i-1], bins.bbox[i]);
            }

            BoundingBox3f bbox_right = bins.bbox[Bins::BIN_COUNT-1], best_bbox_right;
            int64_t best_index = -1;
            float best_cost = (float) INTERSECTION_COST * size;
            float tri_factor = (float) INTERSECTION_COST / node.bbox.getSurfaceArea();

            for (int i=Bins::BIN_COUNT - 2; i >= 0; --i) {
                uint32_t prims_left = bins.counts[i], prims_right = (uint32_t) (end - start) - bins.counts[i];
                float sah_cost = 2.0f * TRAVERSAL_COST +
                                 tri_factor * (prims_left * bbox_left[i].getSurfaceArea() +
                                               prims_right * bbox_right.getSurfaceArea());
                if (sah_cost < best_cost) {
                    best_cost = sah_cost;
                    best_index = i;
                    best_bbox_right = bbox_right;
                }
                bbox_right = BoundingBox3f::merge(bbox_right, bins.bbox[i]);
            }

            if (best_index == -1) {
                /* Could not find a good split plane -- retry with
                   more careful serial code just to be sure.. */
                execute_serially(bvh, node_idx, start, end, temp);
                return nullptr;
            }

            uint32_t left_count = bins.counts[best_index];
            int node_idx_left = node_idx+1;
            int node_idx_right = node_idx+2*left_count;

            bvh.m_nodes[node_idx_left ].bbox = bbox_left[best_index];
            bvh.m_nodes[node_idx_right].bbox = best_bbox_right;
            node.inner.rightChild = node_idx_right;
            node.inner.axis = axis;
            node.inner.flag = 0;

            std::atomic<uint32_t> offset_left(0),
                    offset_right(bins.counts[best_index]);

            tbb::parallel_for(
                    tbb::blocked_range<uint32_t>(0u, size, GRAIN_SIZE),
                    [&](const tbb::blocked_range<uint32_t> &range) {
                        uint32_t count_left = 0, count_right = 0;
                        for (uint32_t i = range.begin(); i != range.end(); ++i) {
                            uint32_t f = start[i];
                            float centroid = bvh.getCentroid(f)[axis];
                            int index = (int) ((centroid - min) * inv_bin_size);
                            (index <= best_index ? count_left : count_right)++;
                        }
                        uint32_t idx_l = offset_left.fetch_add(count_left);
                        uint32_t idx_r = offset_right.fetch_add(count_right);
                        for (uint32_t i = range.begin(); i != range.end(); ++i) {
                            uint32_t f = start[i];
                            float centroid = bvh.getCentroid(f)[axis];
                            int index = (int) ((centroid - min) * inv_bin_size);
                            if (index <= best_index)
                                temp[idx_l++] = f;
                            else
                                temp[idx_r++] = f;
                        }
                    }
            );
            memcpy(start, temp, size * sizeof(uint32_t));
            assert(offset_left == left_count && offset_right == size);

            /* Create an empty parent task */
            tbb::task& c = *new (allocate_continuation()) tbb::empty_task;
            c.set_ref_count(2);

            /* Post right subtree to scheduler */
            BVHBuildTask &b = *new (c.allocate_child())
                    BVHBuildTask(bvh, node_idx_right, start + left_count,
                                 end, temp + left_count);
            spawn(b);

            /* Directly start working on left subtree */
            recycle_as_child_of(c);
            node_idx = node_idx_left;
            end = start + left_count;

            return this;
        }

        /// Single-threaded build function
        static void execute_serially(BVH &bvh, uint32_t node_idx, uint32_t *start, uint32_t *end, uint32_t *temp) {
            BVH::BVHNode &node = bvh.m_nodes[node_idx];
            uint32_t size = (uint32_t) (end - start);
            float best_cost = (float) INTERSECTION_COST * size;
            int64_t best_index = -1, best_axis = -1;
            float *left_areas = (float *) temp;

            /* Try splitting along every axis */
            for (int axis=0; axis<3; ++axis) {
                /* Sort all triangles based on their centroid positions projected on the axis */
                std::sort(start, end, [&](uint32_t f1, uint32_t f2) {
                    return bvh.getCentroid(f1)[axis] < bvh.getCentroid(f2)[axis];
                });

                BoundingBox3f bbox;
                for (uint32_t i = 0; i<size; ++i) {
                    uint32_t f = *(start + i);
                    bbox.expandBy(bvh.getBoundingBox(f));
                    left_areas[i] = (float) bbox.getSurfaceArea();
                }
                if (axis == 0)
                    node.bbox = bbox;

                bbox.reset();

                /* Choose the best split plane */
                float tri_factor = INTERSECTION_COST / node.bbox.getSurfaceArea();
                for (uint32_t i = size-1; i>=1; --i) {
                    uint32_t f = *(start + i);
                    bbox.expandBy(bvh.getBoundingBox(f));

                    float left_area = left_areas[i-1];
                    float right_area = bbox.getSurfaceArea();
                    uint32_t prims_left = i;
                    uint32_t prims_right = size-i;

                    float sah_cost = 2.0f * TRAVERSAL_COST +
                                     tri_factor * (prims_left * left_area +
                                                   prims_right * right_area);

                    if (sah_cost < best_cost) {
                        best_cost = sah_cost;
                        best_index = i;
                        best_axis = axis;
                    }
                }
            }

            if (best_index == -1) {
                /* Splitting does not reduce the cost, make a leaf */
                node.leaf.flag = 1;
                node.leaf.start = (uint32_t) (start - bvh.m_indices.data());
                node.leaf.size  = size;
                return;
            }

            std::sort(start, end, [&](uint32_t f1, uint32_t f2) {
                return bvh.getCentroid(f1)[best_axis] < bvh.getCentroid(f2)[best_axis];
            });

            uint32_t left_count = (uint32_t) best_index;
            uint32_t node_idx_left = node_idx + 1;
            uint32_t node_idx_right = node_idx + 2 * left_count;
            node.inner.rightChild = node_idx_right;
            node.inner.axis = best_axis;
            node.inner.flag = 0;

            execute_serially(bvh, node_idx_left, start, start + left_count, temp);
            execute_serially(bvh, node_idx_right, start+left_count, end, temp + left_count);
        }
    };



    void BVH::addShape(Shape *shape) {
        m_shapes.push_back(shape);
        m_shapeOffset.push_back(m_shapeOffset.back() + CallShape(shape,getPrimitiveCount));
        m_bbox.expandBy(shape->getBoundingBox());
    }

    void BVH::clear() {
        for (auto shape : m_shapes)
            delete shape;
        m_shapes.clear();
        m_shapeOffset.clear();
        m_shapeOffset.push_back(0u);
        m_nodes.clear();
        m_indices.clear();
        m_bbox.reset();
        m_nodes.shrink_to_fit();
        m_shapes.shrink_to_fit();
        m_shapeOffset.shrink_to_fit();
        m_indices.shrink_to_fit();
    }

    void BVH::build() {
        uint32_t size  = getPrimitiveCount();
        if (size == 0)
            return;
        cout << "Constructing a SAH BVH (" << m_shapes.size()
             << (m_shapes.size() == 1 ? " shape, " : " shapes, ")
             << size << " primitives) .. ";
        cout.flush();
        Timer timer;

        /* Conservative estimate for the total number of nodes */
        m_nodes.resize(2*size);
        memset(m_nodes.data(), 0, sizeof(BVHNode) * m_nodes.size());
        m_nodes[0].bbox = m_bbox;
        m_indices.resize(size);

        //if (sizeof(BVHNode) != 32)
        //    throw NoriException("BVH Node is not packed! Investigate compiler settings.");

        for (uint32_t i = 0; i < size; ++i)
            m_indices[i] = i;

        uint32_t *indices = m_indices.data(), *temp = new uint32_t[size];
        BVHBuildTask& task = *new(tbb::task::allocate_root())
                BVHBuildTask(*this, 0u, indices, indices + size , temp);
        tbb::task::spawn_root_and_wait(task);
        delete[] temp;
        std::pair<float, uint32_t> stats = statistics();

        /* The node array was allocated conservatively and now contains
           many unused entries -- do a compactification pass. */
        std::vector<BVHNode> compactified(stats.second);
        std::vector<uint32_t> skipped_accum(m_nodes.size());

        for (int64_t i = stats.second-1, j = m_nodes.size(), skipped = 0; i >= 0; --i) {
            while (m_nodes[--j].isUnused())
                skipped++;
            BVHNode &new_node = compactified[i];
            new_node = m_nodes[j];
            skipped_accum[j] = (uint32_t) skipped;

            if (new_node.isInner()) {
                new_node.inner.rightChild = (uint32_t)
                        (i + new_node.inner.rightChild - j -
                         (skipped - skipped_accum[new_node.inner.rightChild]));
            }
        }
        cout << "done (took " << timer.elapsedString() << " and "
             << memString(sizeof(BVHNode) * m_nodes.size() + sizeof(uint32_t)*m_indices.size())
             << ", SAH cost = " << stats.first
             << ")." << endl;

        m_nodes = std::move(compactified);
        /*Copy stuff over to gp*/


    }

    std::pair<float, uint32_t> BVH::statistics(uint32_t node_idx) const {
        const BVHNode &node = m_nodes[node_idx];
        if (node.isLeaf()) {
            return std::make_pair((float) BVHBuildTask::INTERSECTION_COST * node.leaf.size, 1u);
        } else {
            std::pair<float, uint32_t> stats_left = statistics(node_idx + 1u);
            std::pair<float, uint32_t> stats_right = statistics(node.inner.rightChild);
            float saLeft = m_nodes[node_idx + 1u].bbox.getSurfaceArea();
            float saRight = m_nodes[node.inner.rightChild].bbox.getSurfaceArea();
            float saCur = node.bbox.getSurfaceArea();
            float sahCost =
                    2 * BVHBuildTask::TRAVERSAL_COST +
                    (saLeft * stats_left.first + saRight * stats_right.first) / saCur;
            return std::make_pair(
                    sahCost,
                    stats_left.second + stats_right.second + 1u
            );
        }
    }
    void BVH::transferGpu(NoriObject **objects) {
        for(auto &s:m_shapes){
            s = (Shape *) objects[s->objectId];
        }
        //We cache the indices here for speedup
        for(auto &i : m_indices){
            uint32_t index = i;
            uint32_t shIdx = findShape(index);
            i = index+(shIdx<<20);
            //printf("%p %p %p\n",i,index,shIdx);

        }
        m_shapeOffset.transferGpu();
        m_shapes.transferGpu();
        m_indices.transferGpu();
        m_nodes.transferGpu();

    }

#endif

    __shared__ uint16_t stackB[500*48];

/**will move to gpu **/
    __device__ bool BVH::rayIntersect(const Ray3f &_ray, Intersection &its, bool shadowRay, const Shape* mask) const {

        uint32_t node_idx = 0, stack_idx = 0, leaves[64], leaf_idx = 0;
        uint64_t stack[64];
            its.t = INFINITY;//std::numeric_limits<float>::infinity();
            /* Use an adaptive ray epsilon */

            Ray3f ray(_ray);

            if (ray.mint == Epsilon)
                ray.mint = stdmax(ray.mint, ray.mint * ray.o.array().abs().maxCoeff());

            if (m_nodes.size() == 0 || ray.maxt < ray.mint)
                return false;

            bool foundIntersection = false;

            BVHNode node = m_nodes[0];
            if (!node.bbox.rayIntersect(ray)) {
                return false;
            }

            int lIndex = node_idx+1;
            int rIndex = node.inner.rightChild;
            BVHNode rightNode = m_nodes[node.inner.rightChild];

            uint32_t f = 0;
            while (true) {

                //load children

                BVHNode  leftNode = m_nodes[lIndex];
                const BVHNode  rightNode = m_nodes[rIndex];

                //try to select optimal child node
                float t1,t2;

                bool intersectLeft = leftNode.bbox.rayIntersect(ray);
                bool intersectRight = rightNode.bbox.rayIntersect(ray);


                bool consumeStack = false;
                //we don't intersect  -> load next
                consumeStack = (!intersectLeft && !intersectRight);
                if(!consumeStack){
                    //we intersect both subtrees
                    if(intersectLeft&&intersectRight){
                        //now switch them if left intersects before right
                        //keep stack leaf free, so we push to stack if inner node and add to leaf list if leaf
                        if(rightNode.isLeaf())
                            leaves[leaf_idx++] = rIndex;
                        else {
                            stack[stack_idx++] = ( ((uint64_t)rIndex+1) << 32) + rightNode.inner.rightChild;

                        }
                    }

                    lIndex = 1+ ((intersectLeft)?lIndex:rIndex);
                    rIndex = ((intersectLeft)?leftNode:rightNode).inner.rightChild;
                    bool isLeaf = (intersectLeft)?leftNode.isLeaf():rightNode.isLeaf();
                    if(isLeaf){
                        leaves[leaf_idx++] = lIndex-1;
                        //if found node is a leaf we need to consume the stack
                        consumeStack = true;
                    }

                }
                //as we are guaranteed to not have any inner nodes on  the stack
                if(consumeStack){
                    if(stack_idx==0)
                        break;
                    uint64_t value = *(uint64_t*)&stack[--stack_idx];
                    lIndex = value>>32;
                    rIndex = value&0xFFFFFFFF;
                }
                //as long as all threads have some leafs to work on we are good
                while(__all(leaf_idx>0)){

                    leaf_idx--;
                    BVHNode node = m_nodes.getBVHfirst(leaves[leaf_idx]);
                    for (uint32_t i = node.start(), end = node.end(); i < end; ++i) {
                        uint32_t idx = m_indices[i];
                        uint32_t sIndex = idx >> 20;
                        idx &= 0xFFFFF;

                        //uint32_t sIndex = findShape(idx);
                        const Shape *shape = m_shapes[sIndex];
                        if (mask != NULL && mask != shape) continue;
                        float u, v, t;
                        if (CallShape(shape, rayIntersect, idx, ray, u, v, t)) {
                            if (shadowRay)
                                return true;

                            foundIntersection = true;
                            ray.maxt = its.t = t;
                            its.uv = Point2f(u, v);
                            its.mesh = shape;
                            f = idx;
                        }
                    }

                }

            }
            //these leaves could not be run concurrently
            for(uint32_t l = 0;l<leaf_idx;l++) {
                //loades only the first part of the BVH and leaves the second untouched (saves one tex1Dfetch)
                BVHNode node = m_nodes.getBVHfirst(leaves[l]);
                for (uint32_t i = node.start(), end = node.end(); i < end; ++i) {
                    uint32_t idx = m_indices[i];
                    uint32_t sIndex = idx >> 20;
                    idx &= 0xFFFFF;

                    //uint32_t sIndex = findShape(idx);
                    const Shape *shape = m_shapes[sIndex];
                    if (mask != NULL && mask != shape) continue;
                    float u, v, t;
                    if (CallShape(shape, rayIntersect, idx, ray, u, v, t)) {
                        if (shadowRay)
                            return true;

                        foundIntersection = true;
                        ray.maxt = its.t = t;
                        its.uv = Point2f(u, v);
                        its.mesh = shape;
                        f = idx;
                    }
                }
            }
            if (foundIntersection) {
                CallShape(its.mesh,setHitInformation,f,ray,its);
            }
            return foundIntersection;
        }



    NORI_NAMESPACE_END
/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Wenzel Jakob, Romain Prévost

    Nori is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Nori is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <nori/object.h>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <filesystem/resolver.h>
#include <iomanip>

#if defined(PLATFORM_LINUX)
#include <malloc.h>
#endif

#if defined(PLATFORM_WINDOWS)
#include <windows.h>
#endif

#if defined(PLATFORM_MACOS)
#include <sys/sysctl.h>
#endif

NORI_NAMESPACE_BEGIN

std::string indent(const std::string &string, int amount) {
    /* This could probably be done faster (it's not
       really speed-critical though) */
    std::istringstream iss(string);
    std::ostringstream oss;
    std::string spacer(amount, ' ');
    bool firstLine = true;
    for (std::string line; std::getline(iss, line); ) {
        if (!firstLine)
            oss << spacer;
        oss << line;
        if (!iss.eof())
            oss << endl;
        firstLine = false;
    }
    return oss.str();
}

bool endsWith(const std::string &value, const std::string &ending) {
    if (ending.size() > value.size())
        return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

std::string toLower(const std::string &value) {
    std::string result;
    result.resize(value.size());
    std::transform(value.begin(), value.end(), result.begin(), ::tolower);
    return result;
}

bool toBool(const std::string &str) {
    std::string value = toLower(str);
    if (value == "false")
        return false;
    else if (value == "true")
        return true;
    else
        throw NoriException("Could not parse boolean value \"%s\"", str);
}

int toInt(const std::string &str) {
    char *end_ptr = nullptr;
    int result = (int) strtol(str.c_str(), &end_ptr, 10);
    if (*end_ptr != '\0')
        throw NoriException("Could not parse integer value \"%s\"", str);
    return result;
}

unsigned int toUInt(const std::string &str) {
    char *end_ptr = nullptr;
    unsigned int result = (int) strtoul(str.c_str(), &end_ptr, 10);
    if (*end_ptr != '\0')
        throw NoriException("Could not parse integer value \"%s\"", str);
    return result;
}

float toFloat(const std::string &str) {
    char *end_ptr = nullptr;
    float result = (float) strtof(str.c_str(), &end_ptr);
    if (*end_ptr != '\0')
        throw NoriException("Could not parse floating point value \"%s\"", str);
    return result;
}

size_t vectorSize(const std::string &str) {
    std::vector<std::string> tokens = tokenize(str);
    return tokens.size();
}

Eigen::Vector2f toVector2f(const std::string &str) {
    std::vector<std::string> tokens = tokenize(str);
    if (tokens.size() != 2)
        throw NoriException("Expected 2 values");
    Eigen::Vector2f result;
    for (int i=0; i<2; ++i)
        result[i] = toFloat(tokens[i]);
    return result;
}
Eigen::Vector3f toVector3f(const std::string &str) {
    std::vector<std::string> tokens = tokenize(str);
    if (tokens.size() != 3)
        throw NoriException("Expected 3 values");
    Eigen::Vector3f result;
    for (int i=0; i<3; ++i)
        result[i] = toFloat(tokens[i]);
    return result;
}

std::vector<std::string> tokenize(const std::string &string, const std::string &delim, bool includeEmpty) {
    std::string::size_type lastPos = 0, pos = string.find_first_of(delim, lastPos);
    std::vector<std::string> tokens;

    while (lastPos != std::string::npos) {
        if (pos != lastPos || includeEmpty)
            tokens.push_back(string.substr(lastPos, pos - lastPos));
        lastPos = pos;
        if (lastPos != std::string::npos) {
            lastPos += 1;
            pos = string.find_first_of(delim, lastPos);
        }
    }

    return tokens;
}

std::string timeString(double time, bool precise) {
    if (std::isnan(time) || std::isinf(time))
        return "inf";

    std::string suffix = "ms";
    if (time > 1000) {
        time /= 1000; suffix = "s";
        if (time > 60) {
            time /= 60; suffix = "m";
            if (time > 60) {
                time /= 60; suffix = "h";
                if (time > 12) {
                    time /= 12; suffix = "d";
                }
            }
        }
    }

    std::ostringstream os;
    os << std::setprecision(precise ? 4 : 1)
       << std::fixed << time << suffix;

    return os.str();
}

std::string memString(size_t size, bool precise) {
    double value = (double) size;
    const char *suffixes[] = {
        "B", "KiB", "MiB", "GiB", "TiB", "PiB"
    };
    int suffix = 0;
    while (suffix < 5 && value > 1024.0f) {
        value /= 1024.0f; ++suffix;
    }

    std::ostringstream os;
    os << std::setprecision(suffix == 0 ? 0 : (precise ? 4 : 1))
       << std::fixed << value << " " << suffixes[suffix];

    return os.str();
}

filesystem::resolver *getFileResolver() {
    static filesystem::resolver *resolver = new filesystem::resolver();
    return resolver;
}

__device__ Color3f Color3f::toSRGB() const {
    Color3f result;

    for (int i=0; i<3; ++i) {
        float value = coeff(i);

        if (value <= 0.0031308f)
            result[i] = 12.92f * value;
        else
            result[i] = (1.0f + 0.055f)
                * std::pow(value, 1.0f/2.4f) -  0.055f;
    }

    return result;
}

__device__ Color3f Color3f::toLinearRGB() const {
    Color3f result;

    for (int i=0; i<3; ++i) {
        float value = coeff(i);

        if (value <= 0.04045f)
            result[i] = value * (1.0f / 12.92f);
        else
            result[i] = std::pow((value + 0.055f)
                * (1.0f / 1.055f), 2.4f);
    }

    return result;
}

__device__ bool Color3f::isValid() const {
    for (int i=0; i<3; ++i) {
        float value = coeff(i);
        if (value < 0 || !isfinite(value))
            return false;
    }
    return true;
}

__device__ float Color3f::getLuminance() const {
    return coeff(0) * 0.212671f + coeff(1) * 0.715160f + coeff(2) * 0.072169f;
}

__host__ Transform::Transform(const Eigen::Matrix4f &trafo)
    : m_transform(trafo), m_inverse(trafo.inverse()) { }

std::string Transform::toString() const {
    std::ostringstream oss;
    oss << m_transform.format(Eigen::IOFormat(4, 0, ", ", ";\n", "", "", "[", "]"));
    return oss.str();
}

__device__ Transform Transform::operator*(const Transform &t) const {
    return Transform(m_transform * t.m_transform,
        t.m_inverse * m_inverse);
}

__device__ Vector3f sphericalDirection(float theta, float phi) {
    float sinTheta, cosTheta, sinPhi, cosPhi;

    sincosf(theta, &sinTheta, &cosTheta);
    sincosf(phi, &sinPhi, &cosPhi);

    return Vector3f(
        sinTheta * cosPhi,
        sinTheta * sinPhi,
        cosTheta
    );
}

__device__ Point2f sphericalCoordinates(const Vector3f &v) {
    Point2f result(
        std::acos(v.z()),
        std::atan2(v.y(), v.x())
    );
    if (result.y() < 0)
        result.y() += 2*M_PI;
    return result;
}

__device__ void coordinateSystem(const Vector3f &a, Vector3f &b, Vector3f &c) {
    if (std::abs(a.x()) > std::abs(a.y())) {
        float invLen = 1.0f / std::sqrt(a.x() * a.x() + a.z() * a.z());
        c = Vector3f(a.z() * invLen, 0.0f, -a.x() * invLen);
    } else {
        float invLen = 1.0f / std::sqrt(a.y() * a.y() + a.z() * a.z());
        c = Vector3f(0.0f, a.z() * invLen, -a.y() * invLen);
    }
    b = c.cross(a);
}

__device__ float fresnel(float cosThetaI, float extIOR, float intIOR) {
    float etaI = extIOR, etaT = intIOR;

    if (extIOR == intIOR)
        return 0.0f;

    /* Swap the indices of refraction if the interaction starts
       at the inside of the object */
    if (cosThetaI < 0.0f) {
        float tmp = etaI;
        etaI = etaT;
        etaT = tmp;
        //std::swap(etaI, etaT);
        cosThetaI = -cosThetaI;
    }

    /* Using Snell's law, calculate the squared sine of the
       angle between the normal and the transmitted ray */
    float eta = etaI / etaT,
          sinThetaTSqr = eta*eta * (1-cosThetaI*cosThetaI);

    if (sinThetaTSqr > 1.0f)
        return 1.0f;  /* Total internal reflection! */

    float cosThetaT = std::sqrt(1.0f - sinThetaTSqr);

    float Rs = (etaI * cosThetaI - etaT * cosThetaT)
             / (etaI * cosThetaI + etaT * cosThetaT);
    float Rp = (etaT * cosThetaI - etaI * cosThetaT)
             / (etaT * cosThetaI + etaI * cosThetaT);

    return (Rs * Rs + Rp * Rp) / 2.0f;
}

NORI_NAMESPACE_END
/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Wenzel Jakob, Romain Prévost

    Nori is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Nori is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <nori/camera.h>
//#include <nori/rfilter.h>
#include <nori/warp.h>
#include <Eigen/Geometry>
#include <nori/perspective.h>
NORI_NAMESPACE_BEGIN

/**
 * \brief Perspective camera with depth of field
 *
 * This class implements a simple perspective camera model. It uses an
 * infinitesimally small aperture, creating an infinite depth of field.
 */
  PerspectiveCamera::PerspectiveCamera(const PropertyList &propList) {
        /* Width and height in pixels. Default: 720p */
        m_outputSize.x() = propList.getInteger("width", 1280);
        m_outputSize.y() = propList.getInteger("height", 720);
        m_invOutputSize = m_outputSize.cast<float>().cwiseInverse();

        /* Specifies an optional camera-to-world transformation. Default: none */
        m_cameraToWorld = propList.getTransform("toWorld", Transform());

        /* Horizontal field of view in degrees */
        m_fov = propList.getFloat("fov", 30.0f);

        /* Near and far clipping planes in world-space units */
        m_nearClip = propList.getFloat("nearClip", 1e-4f);
        m_farClip = propList.getFloat("farClip", 1e4f);

        m_rfilter = NULL;
    }

  void PerspectiveCamera::activate()  {
        float aspect = m_outputSize.x() / (float) m_outputSize.y();

        /* Project vectors in camera space onto a plane at z=1:
         *
         *  xProj = cot * x / z
         *  yProj = cot * y / z
         *  zProj = (far * (z - near)) / (z * (far-near))
         *  The cotangent factor ensures that the field of view is 
         *  mapped to the interval [-1, 1].
         */
        float recip = 1.0f / (m_farClip - m_nearClip),
              cot = 1.0f / std::tan(degToRad(m_fov / 2.0f));

        Eigen::Matrix4f perspective;
        perspective <<
            cot, 0,   0,   0,
            0, cot,   0,   0,
            0,   0,   m_farClip * recip, -m_nearClip * m_farClip * recip,
            0,   0,   1,   0;

        /**
         * Translation and scaling to shift the clip coordinates into the
         * range from zero to one. Also takes the aspect ratio into account.
         */
        m_sampleToCamera = Transform( 
            Eigen::DiagonalMatrix<float, 3>(Vector3f(0.5f, -0.5f * aspect, 1.0f)) *
            Eigen::Translation<float, 3>(1.0f, -1.0f/aspect, 0.0f) * perspective).inverse();

        /* If no reconstruction filter was assigned, instantiate a Gaussian filter */
        /*if (!m_rfilter) {
            m_rfilter = static_cast<ReconstructionFilter *>(
                    NoriObjectFactory::createInstance("gaussian", PropertyList()));
            m_rfilter->activate();
        }*/

    }

__device__  Color3f PerspectiveCamera::sampleRay(Ray3f &ray,
            const Point2f &samplePosition,
            const Point2f &apertureSample) const {
        /* Compute the corresponding position on the
           near plane (in local camera space) */
        Point3f nearP = m_sampleToCamera * Point3f(
            samplePosition.x() * m_invOutputSize.x(),
            samplePosition.y() * m_invOutputSize.y(), 0.0f);

        /* Turn into a normalized ray direction, and
           adjust the ray interval accordingly */
        Vector3f d = nearP.normalized();
        float invZ = 1.0f / d.z();

        ray.o = m_cameraToWorld * Point3f(0, 0, 0);
        ray.d = m_cameraToWorld * d;
        ray.mint = m_nearClip * invZ;
        ray.maxt = m_farClip * invZ;
        ray.update();
        return Color3f(1.0f);
    }

    void PerspectiveCamera::addChild(NoriObject *obj)  {
        switch (obj->getClassType()) {
            /*
            case EReconstructionFilter:
                if (m_rfilter)
                    throw NoriException("Camera: tried to register multiple reconstruction filters!");
                m_rfilter = static_cast<ReconstructionFilter *>(obj);
                break;
            */
            default:
                throw NoriException("Camera::addChild(<%s>) is not supported!",
                    classTypeName(obj->getClassType()));
        }
    }

    /// Return a human-readable summary
     std::string PerspectiveCamera::toString() const {
        return tfm::format(
                "PerspectiveCamera[\n"
                        "  cameraToWorld = %s,\n"
                        "  outputSize = %s,\n"
                        "  fov = %f,\n"
                        "  clip = [%f, %f],\n"
                        "  rfilter =  None\n"
                        "]",
                indent(m_cameraToWorld.toString(), 18),
                m_outputSize.toString(),
                m_fov,
                m_nearClip,
                m_farClip
                //indent(m_rfilter->toString())
        );
    }
#ifndef __CUDA_ARCH__
    NORI_REGISTER_CLASS(PerspectiveCamera, "perspective");
#endif

NORI_NAMESPACE_END
#include <nori/integrator.h>
#include <nori/scene.h>
#include <nori/warp.h>
#include <nori/independent.h>
#include <nori/diffuse.h>
#include <nori/arealight.h>
#include <nori/dielectric.h>
//#include <nori/bsdf.h>
//#include <iostream>
#include <nori/pathMisIntegrator.h>
NORI_NAMESPACE_BEGIN


    PathMisIntegrator::PathMisIntegrator(const PropertyList &props) {
        integratorType = EPathMis;
    }

    __device__ Color3f PathMisIntegrator::Li(const Scene *scene, Sampler& sampler, Ray3f &ray,Color3f& t,bool& lastBsdfDelta,float& last_bsdf_pdf) const {

        /* Find the surface that is visible in the requested direction */

		size_t lights = scene->getLights().size();

		// initialize return color
		Color3f li(0.0f);
		float successProb = 0.99;
		//const BSDF* last_bsdf = NULL;

        Intersection its;
        Intersection its_out;

		//bool lastBsdfDelta = true;
		//float last_bsdf_pdf = 1.0f;
		//bool firstRun = true;

		for (int i=0;i<2;i++)
		{
			// trace ray
			if (!scene->rayIntersect(ray, its)){
				//if(firstRun)
				//		asm("exit;");
                t(0) = -1;
                break;
			}
            // we directly hit a light source
            //printf("Color:%f\n",li.x());
			//firstRun = false;
			if (its.mesh->getEmitter() != NULL)
			{
                const Emitter* light = its.mesh->getEmitter();

				EmitterQueryRecord emitterQuery(ray.o, its.p, its.shFrame.n);
				Color3f flight = CallEmitter(light,eval,emitterQuery);

				//BSDFQueryRecord bsdfQuery(shFrame.toLocal(light_dir), its.shFrame.toLocal(cam_dir), ESolidAngle);
				//float bsdf_pdf = bsdf->pdf(bsdfQuery);
				float light_pdf = CallEmitter(light,pdf,emitterQuery);
                //return li;
				float wmat = last_bsdf_pdf / (light_pdf + last_bsdf_pdf);
                //printf("%f\n",light_pdf);
				if (lastBsdfDelta)
					wmat = 1.0f;

				if (wmat > 0.0f)
				{
					//std::cout << wmat << endl;
					li += wmat*t*flight;
				}
                //printf("%f %f %f\n",light_pdf,wmat,flight.x());
			}

     		//successProb = stdmin(0.99f, t.maxCoeff());
     		successProb = 0.95f;
			if (CallSampler(sampler,next1D) > successProb)
			{
                t(0) = -1;
				break;
			}
			else
			{
				t /= successProb;
			}


			// MATS

			const BSDF* bsdf = its.mesh->getBSDF();
			Vector3f wi = -ray.d;

			BSDFQueryRecord bsdfQuery(its.shFrame.toLocal(wi));
			bsdfQuery.measure = ESolidAngle;
			bsdfQuery.uv = its.uv;

            CallBsdf(bsdf, sampleDx, bsdfQuery, its, sampler);
			if (bsdfQuery.probe)
            {
                // bssrdf case
                while (!scene->rayIntersect(bsdfQuery.probeRay, its_out, false, its.mesh))
                    //break;
                    CallBsdf(bsdf, sampleDx, bsdfQuery, its, sampler);

                bsdfQuery.probeDistance = (its_out.p - its.p).norm();
            }
            else
            {
                // brdf case
                its_out = its;
            }
			Color3f f_mat = CallBsdf(bsdf,sample,bsdfQuery, CallSampler(sampler,next2D));
			last_bsdf_pdf = CallBsdf(bsdf,pdf,bsdfQuery);

			lastBsdfDelta = bsdfQuery.isDelta;

			//Vector3f light_dir = its_out.toWorld(bsdfQuery.wo);
			//Vector3f cam_dir = -traceRay.d; // last cam direction
			Vector3f light_dir = its_out.toWorld(bsdfQuery.wo).normalized();
			Vector3f cam_dir = -ray.d; // last cam direction

			// set new wi
			//light_dir.y() += float(0.2);
			ray.d = light_dir;
            ray.o = its_out.p;
            ray.mint = float(1e-3);
            ray.update();
			//printf("ds:%f\n",last_bsdf_pdf);

			// we indirectly might hit a light source
			// EMS PART	
			{
				const Emitter* light = scene->getRandomEmitter(CallSampler(sampler,next1D));
				if (!bsdfQuery.isDelta)
				{
					// prepare light source and create shadowRay 
					EmitterQueryRecord emitterQuery(its_out.p);

					Color3f liLoc = CallEmitter(light,sample,emitterQuery, CallSampler(sampler,next2D));
					Vector3f light_dir = emitterQuery.wi;

					// check if the light actually hits the intersection point
					if (!scene->rayIntersect(emitterQuery.shadowRay))
					{
						//BSDFQueryRecord bsdfQuery(its.shFrame.toLocal(light_dir), its.shFrame.toLocal(cam_dir), ESolidAngle);
						bsdfQuery.wo = its_out.shFrame.toLocal(light_dir);

						// bsdf based color
						float bsdf_pdf = CallBsdf(bsdf,pdf,bsdfQuery);
						float light_pdf = CallEmitter(light,pdf,emitterQuery);
						//printf("%f %f\n",liLoc.x(),f.x());

						float wem = light_pdf / (light_pdf + bsdf_pdf);

						// angle between light ray and surface normal
						float costhetaSurface = light_dir.dot(its_out.shFrame.n);

						//cout << costhetaSurface << endl;
						if (wem > 0)
						{
                            Color3f bsdf_val = CallBsdf(bsdf,eval,bsdfQuery);
							li += t*wem*liLoc*bsdf_val*fabs(costhetaSurface)*lights;
						}
					}
				}
			}


			t *= f_mat;

		}
		return li;
    }

    std::string PathMisIntegrator::toString() const {
        return "PathMisIntegrator[]";
    }

#ifndef __CUDA_ARCH__
    NORI_REGISTER_CLASS(PathMisIntegrator, "path_mis");
#endif
NORI_NAMESPACE_END
/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Wenzel Jakob

    Nori is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Nori is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <cuda_runtime.h>
#include <nori/independent.h>
//#include <nori/block.h>
#include "pcg32.h"

NORI_NAMESPACE_BEGIN

/**
 * Independent sampling - returns independent uniformly distributed
 * random numbers on <tt>[0, 1)x[0, 1)</tt>.
 *
 * This class is essentially just a wrapper around the pcg32 pseudorandom
 * number generator. For more details on what sample generators do in
 * general, refer to the \ref Sampler class.
 */

__host__ size_t Independent::getSize() const {return sizeof(Independent);};
    Independent::Independent(const PropertyList &propList) {
        m_sampleCount = (size_t) propList.getInteger("sampleCount", 1);
    }

__host__ std::string Independent::toString() const  {
        return tfm::format("Independent[]");
    }
#ifndef __CUDA_ARCH__
    NORI_REGISTER_CLASS(Independent, "independent");
#endif




    __device__ void Independent::prepare(Sampler& state, const Point2i &pixel) {
        //run from GPU, this initializes the sampler
        //this gureantees
        state = pixel.x()*10000*pixel.y();
        Independent::next1D(state);
    }

    __device__ void Independent::generate(Sampler& state) { /* No-op for this sampler */ }
    __device__ void Independent::advance(Sampler& state)  { /* No-op for this sampler */ }

    __device__ float Independent::next1D(Sampler& state) {
        Sampler oldstate = state;
        state = oldstate * 0x5851f42d4c957f2dULL;
        uint32_t xorshifted = (uint32_t) (((oldstate >> 18u) ^ oldstate) >> 27u);
        uint32_t rot = (uint32_t) (oldstate >> 59u);
        uint32_t i =  (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
        union {
            uint32_t u;
            float f;
        } x;
        x.u = (i >> 9) | 0x3f800000u;
        return x.f - 1.0f;
    }

    __device__ Point2f Independent::next2D(Sampler& state) {
        return Point2f(
            Independent::next1D(state),
            Independent::next1D(state)
        );
    }

NORI_NAMESPACE_END
#include <nori/dielectric.h>
#include <nori/bsdf.h>
#include <nori/frame.h>
#include <nori/common.h>

NORI_NAMESPACE_BEGIN

/// Ideal dielectric BSDF

__host__ Dielectric::Dielectric(const PropertyList &propList) {
    /* Interior IOR (default: BK7 borosilicate optical glass) */
    m_intIOR = propList.getFloat("intIOR", 1.5046f);

    /* Exterior IOR (default: air) */
    m_extIOR = propList.getFloat("extIOR", 1.000277f);
    
    this->bsdfType = TDielectric;
}

__host__ std::string Dielectric::toString() const 
{
    return tfm::format(
        "Dielectric[\n"
        "  intIOR = %f,\n"
        "  extIOR = %f\n"
        "]",
        m_intIOR, m_extIOR);
}

__device__ Color3f Dielectric::eval(const BSDFQueryRecord &) const  {
    /* Discrete BRDFs always evaluate to zero in Nori */
    return Color3f(0.0f);
}

__device__ float Dielectric::pdf(const BSDFQueryRecord &) const  {
    /* Discrete BRDFs always evaluate to zero in Nori */
    return 0.0f;
}

__device__ Color3f Dielectric::sample(BSDFQueryRecord &bRec, const Point2f &sample) const {
    
    float costhetai = Frame::cosTheta(bRec.wi);

    float extIOR = m_extIOR;
    float intIOR = m_intIOR;
    if (costhetai < 0)
    {
        extIOR = m_intIOR;
        intIOR = m_extIOR;
        //return Color3f(0.0f);
    }
    float fresnel_val = fresnel(fabsf(costhetai), extIOR, intIOR);
    bRec.isDelta = true;

    //float fresnel_val = 0;

    if (sample.x() <= fresnel_val)
    {
        // reflection

        bRec.wo = Vector3f(
            -bRec.wi.x(),
            -bRec.wi.y(),
             bRec.wi.z()
        );
        bRec.measure = EDiscrete;

        /* Relative index of refraction: no change */
        bRec.eta = 1.0f;
    }
    else
    {
        Vector3f n(0,0,1.0f);
        if (costhetai < 0) 
            n = Vector3f(0,0,-1.0f);

        //bRec.wi.normalize();
        float win = bRec.wi.dot(n); 
        float IORrel = extIOR/intIOR;

        // refraction
        bRec.wo = -IORrel*(bRec.wi - win*n) - n*sqrtf(1.0f - IORrel*IORrel*(1.0f - win*win));
        bRec.wo.normalize();

        bRec.measure = EDiscrete;
        bRec.eta = IORrel;

        float IORrelInv = 1.0f/IORrel;
        return Color3f(IORrelInv*IORrelInv);
        //return Color3f(1.0f);
    }

    return Color3f(1.0f);
}

__host__ void Dielectric::gpuTransfer(NoriObject ** objects) {
    //this->m_albedo = (Texture <Color3f> *) objects[m_albedo->objectId];
}



#ifndef __CUDA_ARCH__
    NORI_REGISTER_CLASS(Dielectric, "dielectric");
#endif
NORI_NAMESPACE_END
/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Romain Prévost

    Nori is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Nori is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/
#include <cmath>
#include <nori/emitter.h>
#include <nori/sphere.h>
#include <nori/warp.h>
#include <nori/arealight.h>
#include <nori/mesh.h>
#include <nori/shape.h>
#include <nori/frame.h>
#include <iostream>
#include <cuda_runtime_api.h>

NORI_NAMESPACE_BEGIN

   __host__ AreaEmitter::AreaEmitter(const PropertyList &props){
		this->emitterType = EArea;
        m_radiance = props.getColor("radiance");
   }

   __host__ std::string AreaEmitter::toString() const  {
        return tfm::format(
                "AreaLight[\n"
                "  radiance = %s,\n"
                "]",
                m_radiance.toString());
    }

     __device__ Color3f AreaEmitter::eval(const EmitterQueryRecord & lRec) const {
        //printf("Printf Debugging: %p %f\n",this,m_radiance(0));
        if(!m_shape)
            return 0;
            // throw NoriException("There is no shape attached to this Area light!");
        //printf("%f\n",m_radiance.x());
		// wi points into the light source, n out of the lightsource
		if (lRec.n.dot(lRec.wi) < 0)
		{
			return m_radiance;
		}
		else
		{
			return 0;
		}
    }

    __device__ Color3f AreaEmitter::sample(EmitterQueryRecord & lRec, const Point2f & sample) const {
        if(!m_shape)
            return 0;
            //throw NoriException("There is no shape attached to this Area light!");

		ShapeQueryRecord shapeQuery(lRec.ref);
	    CallShape(m_shape,sampleSurface,shapeQuery, sample);
		//std::cout << shapeQuery.p << std::endl;
		//std::cout << std::endl;

		// points from ref to light emission point
		Vector3f dir = (shapeQuery.p - lRec.ref);

		float dist = dir.norm();
		lRec.wi = dir.normalized();
		lRec.n = shapeQuery.n;
		lRec.p = shapeQuery.p;

		// ray from light emission point to ref (object)
		lRec.shadowRay.o = shapeQuery.p;
		lRec.shadowRay.d = -lRec.wi;
		float delta = 1e-4;
		lRec.shadowRay.mint = delta;
		lRec.shadowRay.maxt = dist-delta;
		lRec.shadowRay.update();

		//std::cout << (lRec.shadowRay(0) - shapeQuery.p) << std::endl << std::endl;
		//std::cout << dist << std::endl;
		//
		//float pdfv = pdf(lRec);

		/*float dist2 = (shapeQuery.p - lRec.ref).squaredNorm();
		float costheta = -lRec.wi.normalized().dot(lRec.n.normalized()); 
		float pdfv = shapeQuery.pdf*dist2/costheta;*/
		float pdfv = pdf(lRec);
		lRec.pdf = pdfv;
		float costheta = -lRec.wi.normalized().dot(lRec.n.normalized());
		if (fabs(pdfv) > float(1e-6))
			return m_radiance/pdfv;
		else return Color3f(0.0f);
    }

   __device__ float AreaEmitter::pdf(const EmitterQueryRecord &lRec) const {
        if(!m_shape)
            return 0;
            // throw NoriException("There is no shape attached to this Area light!");
		
		float dist2 = (lRec.ref - lRec.p).squaredNorm();
		float costheta = -lRec.wi.normalized().dot(lRec.n.normalized());
		if (costheta <= 0)
		{
			return 0.0f;
		}
		else
		{
			ShapeQueryRecord shapeQuery(lRec.ref, lRec.p);

			return CallShape(m_shape,pdfSurface,shapeQuery)*dist2/fabs(costheta);
		}
    }


#ifndef __CUDA_ARCH__
    NORI_REGISTER_CLASS(AreaEmitter, "area")
#endif
NORI_NAMESPACE_END
#include <cuda_runtime_api.h>
#include <vector_functions.h>
#include <nori/scene.h>
#include <nori/camera.h>
#include <device_functions.h>
#include <nori/independent.h>
#include <nori/perspective.h>
#include <nori/integrator.h>
#include <nori/pathMisIntegrator.h>
#include <nori/bvh.h>
#include <chrono>
#include <unistd.h>
class Perspective;
NORI_NAMESPACE_BEGIN

//__shared__ Sampler samplerS[1024];
#define MEDIAN(a,b,c) ((a-b)*(b-c) > -1 ? b : ((a-b)*(a-c) < 1 ? a : c))
__global__ void render_gpu(cudaSurfaceObject_t cuda_data, int width, int height, nori::Scene* scene,Color3f* image,size_t step)
{

    Sampler s;// = samplerS[threadIdx.x * blockDim.y + threadIdx.y]
    int iterations = scene->getSampler()->m_sampleCount;
    for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < width;  x += blockDim.x * gridDim.x) {
       for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < height;  y += blockDim.y * gridDim.y) {


            Independent::prepare(s, Point2i(x+width*step, y+height*step));
            nori::Point2f p((float(x)), (float(y)));
            Ray3f r;
            const nori::Camera *c = scene->getCamera();
            CallCamera(c, sampleRay, r, p, CallSampler(s, next2D));
            /**todo we could use constant memory for the origian of the ray**/
            Point3f rO = r.o;
            Vector3f rD = r.d;
            Integrator *i = scene->getIntegrator();
            //costly but needed, check whether the camera ray hits something, if not this thread won't be useful
            if(!scene->rayIntersect(r))
               asm("exit;");




           Color3f color;
           int in = 0;
           Ray3f nr = r;
           Color3f t = 1;
           float lastBsdfPdf = 1;
           bool  lastBsdfDelta = true;


           while(in<iterations){
                color += CallIntegrator(i, Li, scene, s, nr,t,lastBsdfDelta,lastBsdfPdf);

                if(t(0)<0){
                    nr.o = rO;
                    nr.d = rD;
                    nr.update();
                    t = 1;
                    lastBsdfDelta = true;

                    in++;
                }
            }

            //Color3f color(MEDIAN(colors[0].x(),colors[1].x(),colors[2].x()),MEDIAN(colors[0].y(),colors[1].y(),colors[2].y()),MEDIAN(colors[0].z(),colors[1].z(),colors[2].z()));
            color /= iterations;
            //if (step) {
            //    color = (color - image[gIn]) / (step + 1) + image[gIn];
           // }

            Color3f colorO = color.toSRGB();
            surf2Dwrite(make_float4(colorO(0), colorO(1), colorO(2), 1.0f),
                        cuda_data, (int) sizeof(float4) * x, height - y,
                        cudaBoundaryModeClamp);

            //image(gIn).x() = color.x();
            //image[gIn] = color;
        }
    }
}

/**
 *
 * @param resource
 * @param w
 * @param h
 * @param scene  scene object on GPU memory
 * @param sampler  Sampler object on CPU memory
 */
static Color3f* image = nullptr;
//static nori::Independent* gpuSampler = nullptr;
static size_t step = 0;
static  std::chrono::milliseconds startTime;

void render_scene(cudaSurfaceObject_t resource, int w, int h,nori::Scene *scene)
{



    int blockSize;
    int gridSize;

    cudaOccupancyMaxPotentialBlockSize(&gridSize,&blockSize,render_gpu,0,w*h);

    //we want to render 2D blocks not lines
    int blockW = sqrt(blockSize);
    int blockH = blockSize/blockW;
    dim3 block(blockW,blockH);
    int gridSizeW = (w + blockW - 1) / blockW;
    int gridSizeH = (h + blockH - 1) / blockH;
    dim3 grid(gridSizeW,gridSizeH);
    //At the moment we do not use the given Sampler but simply independent
    if(!image) {
        startTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
        cudaMalloc((void **) &image, w * h * sizeof(Color3f));
        cudaMemset(image, 0, w * h * sizeof(Color3f));
    }


    //number of kernel runs
    const int sCount = 1;
    //int block = 64;
    //int grid  = (w*h + block - 1) / block;
    if(step<sCount) {
        nori::render_gpu << < grid,block >> > (resource, w, h, scene,  image, step);
        std::cout<<cudaGetErrorString(cudaGetLastError())<<std::endl;
        std::cout<<step<<std::endl;
        step++;
    }else if(sCount == step){
       if(cudaSuccess==cudaStreamQuery(0)){
            cudaDeviceSynchronize();
            auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()) - startTime;
            std::cout << "finished: Took" << float(diff.count())/1000  << " seconds!"<< std::endl ;
            step++;
        //    exit(0);
       }else{
           sleep(1);
       }
    }

}

NORI_NAMESPACE_END


// workaround issue between gcc >= 4.7 and cuda 5.5
#if (defined __GNUC__) && (__GNUC__>4 || __GNUC_MINOR__>=7)
  #undef _GLIBCXX_ATOMIC_BUILTINS
  #undef _GLIBCXX_USE_INT128
#endif

#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX
#define EIGEN_TEST_FUNC cuda_basic
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

#include "main.h"
#include "cuda_common.h"

#include <Eigen/Eigenvalues>

// struct Foo{
//   EIGEN_DEVICE_FUNC
//   void operator()(int i, const float* mats, float* vecs) const {
//     using namespace Eigen;
//   //   Matrix3f M(data);
//   //   Vector3f x(data+9);
//   //   Map<Vector3f>(data+9) = M.inverse() * x;
//     Matrix3f M(mats+i/16);
//     Vector3f x(vecs+i*3);
//   //   using std::min;
//   //   using std::sqrt;
//     Map<Vector3f>(vecs+i*3) << x.minCoeff(), 1, 2;// / x.dot(x);//(M.inverse() *  x) / x.x();
//     //x = x*2 + x.y() * x + x * x.maxCoeff() - x / x.sum();
//   }
// };

template<typename T>
struct coeff_wise {
  EIGEN_DEVICE_FUNC
  void operator()(int i, const typename T::Scalar* in, typename T::Scalar* out) const
  {
    using namespace Eigen;
    T x1(in+i);
    T x2(in+i+1);
    T x3(in+i+2);
    Map<T> res(out+i*T::MaxSizeAtCompileTime);
    
    res.array() += (in[0] * x1 + x2).array() * x3.array();
  }
};

template<typename T>
struct redux {
  EIGEN_DEVICE_FUNC
  void operator()(int i, const typename T::Scalar* in, typename T::Scalar* out) const
  {
    using namespace Eigen;
    int N = 6;
    T x1(in+i);
    out[i*N+0] = x1.minCoeff();
    out[i*N+1] = x1.maxCoeff();
    out[i*N+2] = x1.sum();
    out[i*N+3] = x1.prod();
//     out[i*N+4] = x1.colwise().sum().maxCoeff();
//     out[i*N+5] = x1.rowwise().maxCoeff().sum();
  }
};

template<typename T1, typename T2>
struct prod {
  EIGEN_DEVICE_FUNC
  void operator()(int i, const typename T1::Scalar* in, typename T1::Scalar* out) const
  {
    using namespace Eigen;
    typedef Matrix<typename T1::Scalar, T1::RowsAtCompileTime, T2::ColsAtCompileTime> T3;
    T1 x1(in+i);
    T2 x2(in+i+1);
    Map<T3> res(out+i*T3::MaxSizeAtCompileTime);
    res += in[i] * x1 * x2;
  }
};

template<typename T1, typename T2>
struct diagonal {
  EIGEN_DEVICE_FUNC
  void operator()(int i, const typename T1::Scalar* in, typename T1::Scalar* out) const
  {
    using namespace Eigen;
    T1 x1(in+i);
    Map<T2> res(out+i*T2::MaxSizeAtCompileTime);
    res += x1.diagonal();
  }
};

template<typename T>
struct eigenvalues {
  EIGEN_DEVICE_FUNC
  void operator()(int i, const typename T::Scalar* in, typename T::Scalar* out) const
  {
    using namespace Eigen;
    typedef Matrix<typename T::Scalar, T::RowsAtCompileTime, 1> Vec;
    T M(in+i);
    Map<Vec> res(out+i*Vec::MaxSizeAtCompileTime);
    T A = M*M.adjoint();
    SelfAdjointEigenSolver<T> eig;
    eig.computeDirect(M);
    res = eig.eigenvalues();
  }
};

void test_cuda_basic()
{
  ei_test_init_cuda();
  
  int nthreads = 100;
  Eigen::VectorXf in, out;
  
  #ifndef __CUDA_ARCH__
  int data_size = nthreads * 16;
  in.setRandom(data_size);
  out.setRandom(data_size);
  #endif
  
  CALL_SUBTEST( run_and_compare_to_cuda(coeff_wise<Vector3f>(), nthreads, in, out) );
  CALL_SUBTEST( run_and_compare_to_cuda(coeff_wise<Array44f>(), nthreads, in, out) );
  
  CALL_SUBTEST( run_and_compare_to_cuda(redux<Array4f>(), nthreads, in, out) );
  CALL_SUBTEST( run_and_compare_to_cuda(redux<Matrix3f>(), nthreads, in, out) );
  
  CALL_SUBTEST( run_and_compare_to_cuda(prod<Matrix3f,Matrix3f>(), nthreads, in, out) );
  CALL_SUBTEST( run_and_compare_to_cuda(prod<Matrix4f,Vector4f>(), nthreads, in, out) );
  
  CALL_SUBTEST( run_and_compare_to_cuda(diagonal<Matrix3f,Vector3f>(), nthreads, in, out) );
  CALL_SUBTEST( run_and_compare_to_cuda(diagonal<Matrix4f,Vector4f>(), nthreads, in, out) );
  
  CALL_SUBTEST( run_and_compare_to_cuda(eigenvalues<Matrix3f>(), nthreads, in, out) );
  CALL_SUBTEST( run_and_compare_to_cuda(eigenvalues<Matrix2f>(), nthreads, in, out) );

}
