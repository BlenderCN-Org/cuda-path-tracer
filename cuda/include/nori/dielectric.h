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

#ifndef NORIGL_DIELECTRIC_H
#define NORIGL_DIELECTRIC_H

#include <nori/bsdf.h>
#include <nori/frame.h>
#include <nori/common.h>


NORI_NAMESPACE_BEGIN

/// Ideal dielectric BSDF
class Dielectric : public BSDF {
public:
    __host__ Dielectric(const PropertyList &propList);

    /// Evaluate the BRDF model
    __device__ Color3f eval(const BSDFQueryRecord &bRec) const  ;

    /// Compute the density of \ref sample() wrt. solid angles
    __device__ float pdf(const BSDFQueryRecord &bRec) const  ;
    /// Draw a a sample from the BRDF model
    __device__ Color3f sample(BSDFQueryRecord &bRec, const Point2f &sample) const ;
    // sample ray close to intersection for bssrdf evaluation
    __device__ void sampleDx(BSDFQueryRecord &bRec, const Intersection& its, Sampler& sampler) const {};

    __host__ virtual void gpuTransfer(NoriObject **) override;
    __host__ virtual std::string toString() const override;
    __host__ virtual size_t getSize() const  override  {return sizeof(Dielectric);}

private:
    float m_intIOR, m_extIOR;
};

NORI_NAMESPACE_END
#endif
