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

#if !defined(__NORI_BSDF_H)
#define __NORI_BSDF_H

#include <nori/object.h>
#include <host_defines.h>

NORI_NAMESPACE_BEGIN

/**
 * \brief Convenience data structure used to pass multiple
 * parameters to the evaluation and sampling routines in \ref BSDF
 */
struct BSDFQueryRecord {
    /// Incident direction (in the local frame)
    Vector3f wi;

    /// Outgoing direction (in the local frame)
    Vector3f wo;

    /// Relative refractive index in the sampled direction
    float eta;

    /// Measure associated with the sample
    EMeasure measure;

    /// Create a new record for sampling the BSDF
    __device__ __host__ BSDFQueryRecord(const Vector3f &wi)
        : wi(wi), measure(EUnknownMeasure) { }

    /// Create a new record for querying the BSDF
    __device__ __host__ BSDFQueryRecord(const Vector3f &wi,
            const Vector3f &wo, EMeasure measure)
        : wi(wi), wo(wo), measure(measure) { }


    /// Additional information possibly needed by the BSDF
    /// UV associated with the point
    Point2f uv;
    /// Point associated with the point
    Point3f p;

    bool isDelta = false;

    // prope ray to sample bssrdfs
    bool probe = false;
    Ray3f probeRay;
    float probeDistance; // distance of probe
};

/**
 * \brief Superclass of all bidirectional scattering distribution functions
 */
class BSDF : public NoriObject {
public:
    virtual size_t getSize() const override  {return sizeof(BSDF);}

    enum bsdfType {
        TDiffuse,
        TDielectric,
        TSubsurface,
        TMirror
    };
    bsdfType  bsdfType;
    /**
     * \brief Sample the BSDF and return the importance weight (i.e. the
     * value of the BSDF * cos(theta_o) divided by the probability density
     * of the sample with respect to solid angles).
     *
     * \param bRec    A BSDF query record
     * \param sample  A uniformly distributed sample on \f$[0,1]^2\f$
     *
     * \return The BSDF value divided by the probability density of the sample
     *         sample. The returned value also includes the cosine
     *         foreshortening factor associated with the outgoing direction,
     *         when this is appropriate. A zero value means that sampling
     *         failed.
     */
     //virtual Color3f sample(BSDFQueryRecord &bRec, const Point2f &sample) const = 0;

	 //bool isDelta = false;//() const = 0;

    /**
     * \brief Evaluate the BSDF for a pair of directions and measure
     * specified in \code bRec
     *
     * \param bRec
     *     A record with detailed information on the BSDF query
     * \return
     *     The BSDF value, evaluated for each color channel
     */
    //virtual Color3f eval(const BSDFQueryRecord &bRec) const = 0;

    /**
     * \brief Compute the probability of sampling \c bRec.wo
     * (conditioned on \c bRec.wi).
     *
     * This method provides access to the probability density that
     * is realized by the \ref sample() method.
     *
     * \param bRec
     *     A record with detailed information on the BSDF query
     *
     * \return
     *     A probability/density value expressed with respect
     *     to the specified measure
     */

    //virtual float pdf(const BSDFQueryRecord &bRec) const = 0;

    /**
     * \brief Return the type of object (i.e. Mesh/BSDF/etc.)
     * provided by this instance
     * */

    virtual EClassType getClassType() const override { return EBSDF; }

    /**
     * \brief Return whether or not this BRDF is diffuse. This
     * is primarily used by photon mapping to decide whether
     * or not to store photons on a surface
     */
   bool isDiffuse = false;
};


/*
 * Need to add all BSDFs here for usage because no (working) virtual function
 */

#define CallBsdf(object,function,...) CallCuda3(BSDF,bsdfType,object,function,const,  TDiffuse,Diffuse, TDielectric,Dielectric,  TSubsurface,Subsurface,   __VA_ARGS__)

NORI_NAMESPACE_END

#endif /* __NORI_BSDF_H */