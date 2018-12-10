#ifndef NORIGL_SUBSURFACE_H
#define NORIGL_SUBSURFACE_H

#include <nori/proplist.h>
#include <nori/bsdf.h>
#include <nori/frame.h>
#include <nori/warp.h>
#include <nori/texture.h>
#include <nori/common.h>
#include <nori/shape.h>

NORI_NAMESPACE_BEGIN
class Subsurface : public BSDF {
public:

    __host__ Subsurface(const PropertyList &propList);
    __host__ virtual ~Subsurface() override;
    __host__ virtual void addChild(NoriObject* obj) override;
    __host__ virtual void activate() override;
    __host__ virtual void gpuTransfer(NoriObject **) override;
    __host__ virtual std::string toString() const override;
    __host__ virtual size_t getSize() const  override  {return sizeof(Subsurface);}

    /// Evaluate the BRDF model
    __device__ Color3f eval(const BSDFQueryRecord &bRec) const  ;
    /// Compute the density of \ref sample() wrt. solid angles
    __device__ float pdf(const BSDFQueryRecord &bRec) const  ;
    /// Draw a a sample from the BRDF model
    __device__ Color3f sample(BSDFQueryRecord &bRec, const Point2f &sample) const ;
    // sample ray close to intersection for bssrdf evaluation
    __device__ void sampleDx(BSDFQueryRecord &bRec, const Intersection& its, Sampler& sampler) const;

private:

    __device__ float colorNorm(const Color3f& c) const;
    //__device__ Vector3f expVec(const Vector3f& v) const;
    __device__ float diffuseScatterApproxCDF(float r) const;
    __device__ float diffuseScatterApproxCDFDerivative(float r) const;
    __device__ Point2f diffuseScatterPDFsample(const Point2f& sample2) const;
    __device__ float diffuseScatterPDF(const Point2f& p) const;
    __device__ Color3f diffuseScatteringApprox(const BSDFQueryRecord &bRec) const;
    __device__ float fresnelTransmission(const BSDFQueryRecord &bRec) const;


    float m_extIOR;
    float m_intIOR;
    Texture<Color3f> * m_albedo;
    Color3f dmfp;
    Color3f s;
};
NORI_NAMESPACE_END
#endif //NORIGL_SUBSURFACE_H
