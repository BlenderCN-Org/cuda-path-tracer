#ifndef NORIGL_INDEPENDENT_H
#define NORIGL_INDEPENDENT_H
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
#include <nori/common.h>
#include <cuda_runtime.h>
//#include <nori/block.h>
#include "pcg32.h"
#include <curand.h>
#include <curand_kernel.h>


NORI_NAMESPACE_BEGIN

/**
 * Independent sampling - returns independent uniformly distributed
 * random numbers on <tt>[0, 1)x[0, 1)</tt>.
 *
 * This class is essentially just a wrapper around the pcg32 pseudorandom
 * number generator. For more details on what sample generators do in
 * general, refer to the \ref Sampler class.
 */
class Independent:  public NoriObject {
    public:

    __host__ virtual EClassType getClassType() const override { return ESampler; }
    __host__ virtual size_t getSize() const  override;
    __host__ virtual std::string toString() const override;
    __host__ Independent(const PropertyList &propList);

    __device__ static void  prepare(Sampler& s,const Point2i &pixel) ;

    __device__ static void generate(Sampler& s);
    __device__ static void advance(Sampler& s) ;

    __device__ static float next1D(Sampler& s);

    __device__ static Point2f next2D(Sampler& s);
   size_t m_sampleCount;
};

#define CallSampler(object,function,...) Independent::function(object)

NORI_NAMESPACE_END

#endif //NORIGL_INDEPENDENT_H
