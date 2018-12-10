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
        state = oldstate * 0x5851f42d4c957f2dULL + 1;
        uint32_t xorshifted = (uint32_t) (((oldstate >> 18u) ^ oldstate) >> 27u);
        uint32_t rot = (uint32_t) (oldstate >> 59u);
        uint32_t i =  (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
        union {
            uint32_t u;
            float f;
        } x;
        x.u = (i >> 9) | 0x3f800000u;
        float v = x.f - 1.0f;

        return v;

    }

    __device__ Point2f Independent::next2D(Sampler& state) {
        return Point2f(
            Independent::next1D(state),
            Independent::next1D(state)
        );
    }

NORI_NAMESPACE_END
