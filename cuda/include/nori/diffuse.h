//
// Created by lukas on 20.11.17.
//

#ifndef NORIGL_DIFFUSE_H_H
#define NORIGL_DIFFUSE_H_H

#include <nori/proplist.h>
#include <nori/bsdf.h>
#include <nori/frame.h>
#include <nori/warp.h>
#include <nori/texture.h>
#include <nori/common.h>

NORI_NAMESPACE_BEGIN
class Diffuse : public BSDF {
public:

    __host__ Diffuse(const PropertyList &propList);
    __host__ virtual ~Diffuse() override;
    __host__ virtual void  addChild(NoriObject* obj) override;
    __host__ virtual void activate() override;
    __host__ virtual void gpuTransfer(NoriObject **) override;
    __host__ virtual std::string toString() const override;
    __host__ virtual size_t getSize() const  override  {return sizeof(Diffuse);}

    /// Evaluate the BRDF model
    __device__ Color3f eval(const BSDFQueryRecord &bRec) const  ;
    /// Compute the density of \ref sample() wrt. solid angles
    __device__ float pdf(const BSDFQueryRecord &bRec) const  ;
    /// Draw a a sample from the BRDF model
    __device__ Color3f sample(BSDFQueryRecord &bRec, const Point2f &sample) const ;
    __device__ void sampleDx(BSDFQueryRecord &bRec, const Intersection& its, Sampler& sampler) const {};

private:
    Texture<Color3f> * m_albedo;
};
NORI_NAMESPACE_END
#endif //NORIGL_DIFFUSE_H_H
