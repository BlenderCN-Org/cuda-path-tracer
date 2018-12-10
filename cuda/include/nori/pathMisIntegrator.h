//
// Created by lukas on 26.11.17.
//

#ifndef NORIGL_PATHMISINTEGRATOR_H_H
#define NORIGL_PATHMISINTEGRATOR_H_H
#include <nori/integrator.h>
#include <nori/scene.h>
#include <nori/warp.h>
#include <nori/independent.h>
#include <nori/diffuse.h>
#include <nori/arealight.h>
//#include <nori/bsdf.h>
//#include <iostream>
#include <nori/common.h>
#include <nori/integrator.h>
#include <nori/common.h>
NORI_NAMESPACE_BEGIN

class PathMisIntegrator : public Integrator {
    public:
        float maxLen;
        virtual size_t getSize() const override {return sizeof(PathMisIntegrator);}
        PathMisIntegrator(const PropertyList &props);
        __device__ Color3f Li(const Scene *scene, Sampler& sampler, Ray3f &_ray,Color3f& t,bool& lastBsdfDelta,float& last_bsdf_pdf) const;
        std::string toString() const;
};
NORI_NAMESPACE_END
#endif //NORIGL_PATHMISINTEGRATOR_H_H
