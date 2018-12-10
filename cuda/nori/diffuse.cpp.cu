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
