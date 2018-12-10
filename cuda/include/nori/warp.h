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

#if !defined(__NORI_WARP_H)
#define __NORI_WARP_H

#include <nori/common.h>

NORI_NAMESPACE_BEGIN

/// A collection of useful warping functions for importance sampling
namespace Warp{

    __host__ __device__   Vector3f squareToUniformCylinder(const Point2f& sample);

    /// Uniformly sample a vector on the unit hemisphere with respect to solid angles (naive implementation)
    //device only as sampler does not work on cpu currently
    __device__   Vector3f sampleUniformHemisphere(Sampler sampler, const Normal3f &northPole);

    /// Dummy warping function: takes uniformly distributed points in a square and just returns them
    __host__ __device__   Point2f squareToUniformSquare(const Point2f &sample);

    /// Probability density of \ref squareToUniformSquare()
    __host__ __device__   float squareToUniformSquarePdf(const Point2f &p);

    /// Uniformly sample a vector on a 2D disk with radius 1, centered around the origin
    __host__ __device__   Point2f squareToUniformDisk(const Point2f &sample);

    /// Probability density of \ref squareToUniformDisk()
    __host__ __device__   float squareToUniformDiskPdf(const Point2f &p);

    /// Uniformly sample a vector on the unit sphere with respect to solid angles
    __host__ __device__   Vector3f squareToUniformSphere(const Point2f &sample);

    /// Probability density of \ref squareToUniformSphere()
    __host__ __device__   float squareToUniformSpherePdf(const Vector3f &v);

    /**
     * \brief Uniformly sample a vector on a spherical cap around (0, 0, 1)
     *
     * A spherical cap is the subset of a unit sphere whose directions
     * make an angle of less than 'theta' with the north pole. This function
     * expects the cosine of 'theta' as a parameter.
     */
    __host__ __device__   Vector3f squareToUniformSphereCap(const Point2f &sample, float cosThetaMax);

    /// Probability density of \ref squareToUniformSphereCap()
    __host__ __device__   float squareToUniformSphereCapPdf(const Vector3f &v, float cosThetaMax);

    /// Uniformly sample a vector on the unit hemisphere around the pole (0,0,1) with respect to solid angles
    __host__ __device__   Vector3f squareToUniformHemisphere(const Point2f &sample);

    /// Probability density of \ref squareToUniformHemisphere()
    __host__ __device__   float squareToUniformHemispherePdf(const Vector3f &v);

    /// Uniformly sample a vector on the unit hemisphere around the pole (0,0,1) with respect to projected solid angles
    __host__ __device__   Vector3f squareToCosineHemisphere(const Point2f &sample);

    /// Probability density of \ref squareToCosineHemisphere()
    __host__ __device__   float squareToCosineHemispherePdf(const Vector3f &v);

    /// Warp a uniformly distributed square sample to a Beckmann distribution * cosine for the given 'alpha' parameter
    __host__ __device__   Vector3f squareToBeckmann(const Point2f &sample, float alpha);

    /// Probability density of \ref squareToBeckmann()
    __host__ __device__   float squareToBeckmannPdf(const Vector3f &m, float alpha);


    __device__   Vector3f squareToUniformTriangle(const Point2f &sample);

};

NORI_NAMESPACE_END

#endif /* __NORI_WARP_H */
