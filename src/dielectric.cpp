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

#include <nori/bsdf.h>
#include <nori/frame.h>
#include <nori/common.h>

NORI_NAMESPACE_BEGIN

/// Ideal dielectric BSDF
class Dielectric : public BSDF {
public:
    Dielectric(const PropertyList &propList) {
        /* Interior IOR (default: BK7 borosilicate optical glass) */
        m_intIOR = propList.getFloat("intIOR", 1.5046f);

        /* Exterior IOR (default: air) */
        m_extIOR = propList.getFloat("extIOR", 1.000277f);
    }

	virtual bool isDelta() const override { return true; }  

    virtual Color3f eval(const BSDFQueryRecord &) const override {
        /* Discrete BRDFs always evaluate to zero in Nori */
        return Color3f(0.0f);
    }

    virtual float pdf(const BSDFQueryRecord &) const override {
        /* Discrete BRDFs always evaluate to zero in Nori */
        return 0.0f;
    }

    virtual Color3f sample(BSDFQueryRecord &bRec, const Intersection& its, const Point2f &sample, const Point2f &sample2) const override {
		
		float costhetai = Frame::cosTheta(bRec.wi);

		float extIOR = m_extIOR;
		float intIOR = m_intIOR;
		if (costhetai < 0)
		{
			extIOR = m_intIOR;
			intIOR = m_extIOR;
			//return Color3f(0.0f);
		}
		float fresnel_val = fresnel(fabs(costhetai), extIOR, intIOR);
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
			bRec.wo = -IORrel*(bRec.wi - win*n) - n*sqrt(1.0 - IORrel*IORrel*(1.0 - win*win));
			bRec.wo.normalize();

			bRec.measure = EDiscrete;
			bRec.eta = IORrel;

			float IORrelInv = 1.0f/IORrel;
        	return Color3f(IORrelInv*IORrelInv);
        	//return Color3f(1.0f);
		}

        return Color3f(1.0f);
    }

    virtual std::string toString() const override {
        return tfm::format(
            "Dielectric[\n"
            "  intIOR = %f,\n"
            "  extIOR = %f\n"
            "]",
            m_intIOR, m_extIOR);
    }
private:
    float m_intIOR, m_extIOR;
};

NORI_REGISTER_CLASS(Dielectric, "dielectric");
NORI_NAMESPACE_END
