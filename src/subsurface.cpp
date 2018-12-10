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

#include <nori/bsdf.h>
#include <nori/frame.h>
#include <nori/common.h>
#include <nori/warp.h>
#include <nori/texture.h>

NORI_NAMESPACE_BEGIN

/**
 * \brief Subsurface / Lambertian BRDF model
 */
class Subsurface : public BSDF {
public:
    Subsurface(const PropertyList &propList) : m_albedo(nullptr) {
        if(propList.has("albedo")) {
            PropertyList l;
            l.setColor("value", propList.getColor("albedo"));
            m_albedo = static_cast<Texture<Color3f> *>(NoriObjectFactory::createInstance("constant_color", l));
        }

        m_intIOR = propList.getFloat("intIOR", 1.3046f);
        m_extIOR = propList.getFloat("extIOR", 1.000277f);

        //sigma_a = propList.getColor("sigma_a", 0.01f);
        //sigma_s = propList.getColor("sigma_s", 0.1f);

        sigma_t = propList.getColor("sigma_t", 0.01f);
        dmfp = propList.getColor("dmfp", 0.1f);

        //m_intIOR = propList.getFloat("intIOR", 1.0);
        //m_extIOR = propList.getFloat("extIOR", 1.0);
    }

    virtual ~Subsurface() {
        delete m_albedo;
    }


	virtual bool isDelta() const override { return false; }  

    /// Add texture for the albedo
    virtual void addChild(NoriObject *obj) override {
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
                throw NoriException("Subsurface::addChild(<%s>) is not supported!",
                                    classTypeName(obj->getClassType()));
        }
    }

    virtual void activate() override {
        if(!m_albedo) {
            PropertyList l;
            l.setColor("value", Color3f(0.5f));
            m_albedo = static_cast<Texture<Color3f> *>(NoriObjectFactory::createInstance("constant_color", l));
            m_albedo->activate();
        }

        // approx (https://graphics.pixar.com/library/ApproxBSSRDF/paper.pdf)
        //s = 3.5f + 100.0f*pow((albedo - 0.33f), 4); // equation 8
        //s = 1.9f - albedo + 3.5f*pow((albedo - 0.8f), 2); // equation 6
        //s = 1.85f - albedo + 7.0f*abs(pow((albedo - 0.8f), 3)); // equation 5
        s = 1.0f; //(set s=1 is equal to: http://blog.selfshadow.com/publications/s2015-shading-course/burley/s2015_pbs_disney_bsdf_notes.pdf)
    }

    float colorNorm(const Color3f& c) const
    {
        return sqrt(c.x()*c.x() + c.y()*c.y() + c.z()*c.z());
    }

    /// Evaluate the BRDF model
    virtual Color3f eval(const BSDFQueryRecord &bRec) const override {

        //we want: return diffuseScatteringApprox() / bRec.dxPdf(); -> R(r) cancels and this gives:
        float F = fresnelTransmission(bRec);
        return F*m_albedo->eval(bRec.uv)*INV_PI; // bRec.dxAxisPdf;
    }

    float diffuseScatterApproxCDF(float r) const
    {
        float rsd = -r*colorNorm(s)/colorNorm(dmfp);
        return 1.0f - 0.25f*exp(rsd)-0.75f*exp(rsd/3.0f);
    }

    float diffuseScatterApproxCDFDerivative(float r) const
    {
        float sd = colorNorm(s)/colorNorm(dmfp);
        float rsd = -r*sd;
        return sd * 0.25f* (exp(rsd) + exp(rsd/3.0f));
    }

    Point2f diffuseScatterPDFsample(const Point2f& sample2) const
    {
        float sample = sample2.x();

        // newton find inverse of CDF
        float err = 1.0f;
        float sdInv = colorNorm(dmfp)/colorNorm(s);
        float r0 = -log((1.0f - sample) * 4.0/3.0)*3.0f*sdInv;
        float r1 = -log((1.0f - sample) * 4.0)*3.0f*sdInv;
        
        float err0 = fabs(diffuseScatterApproxCDF(r0) - sample);;
        float err1 = fabs(diffuseScatterApproxCDF(r1) - sample);

        if (err0 < err1)  r1=r0;
        else r1 = r0;

        //(cout << "|phi - cdf(r)|_0a=" << fabs(diffuseScatterApproxCDF(r0) - sample) << endl;
        //cout << "|phi - cdf(r)|_0b=" << fabs(diffuseScatterApproxCDF(r1) - sample) << endl;

        for (int i=0; i<3; ++i)
        {
            float f = (diffuseScatterApproxCDF(r0) - sample);
            float df = diffuseScatterApproxCDFDerivative(r0);
            r1 = r0 - f/df;
            r0 = r1;
        }

        //cout << "|phi - cdf(r)|_1=" << fabs(diffuseScatterApproxCDF(r0) - sample) << endl;
		//cout << r0 << endl;
        float theta = 2.0f*M_PI*sample2.y();
        return Point2f(r0*cos(theta), r0*sin(theta));
    }

    float diffuseScatterPDF(const Point2f& p) const
    {
        float sn = colorNorm(s);
        float dmfpn = colorNorm(dmfp);
        float r = p.norm();

        return sn*(exp(-sn*r/dmfpn) + exp(-sn*r/(3.0f*dmfpn)))/(8.0f*M_PI*dmfpn*r);
        //return diffuseScatterApproxCDFDerivative(r) / (2*M_PI*r);
    }

    Color3f diffuseScatteringApprox(const BSDFQueryRecord &bRec) const
    {
        // approx (https://graphics.pixar.com/library/ApproxBSSRDF/paper.pdf equation 3)
        if (bRec.measure != ESolidAngle
            || Frame::cosTheta(bRec.wi) <= 0
            || Frame::cosTheta(bRec.wo) <= 0)
			return 0.0f;

        float r = bRec.r;
        Color3f Rr = m_albedo->eval(bRec.uv)*s*(exp(-s*r/dmfp) + exp(-s*r/(3.0f*dmfp)))/(8.0f*M_PI*dmfp*r);
        float F = fresnelTransmission(bRec);
        return F*Rr*INV_PI;
    }

    float fresnelTransmission(const BSDFQueryRecord &bRec) const
    {
        float Ft_i = 1.0f-fresnel(Frame::cosTheta(bRec.wi), m_extIOR, m_intIOR);
        float Ft_o = 1.0f-fresnel(Frame::cosTheta(bRec.wo), m_extIOR, m_intIOR);
        return Ft_i*Ft_o;
    }

    /// Compute the density of \ref sample() wrt. solid angles
    virtual float pdf(const BSDFQueryRecord &bRec) const override {
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

        return INV_PI * Frame::cosTheta(bRec.wo); //* bRec.dxAxisPdf;
    }

    virtual void sampleDx(BSDFQueryRecord &bRec, const Intersection& its, Sampler& sampler) const override 
    {
        // scatter ray test
        Ray3f traceRay;

        float thres = sampler.next1D();

		float k = 0.5f / 4.0f;

		if (thres < 0.5f)
			traceRay.d = -its.shFrame.n; // to surface parallel to normal
		else if (thres < 0.5f + k)
			traceRay.d = its.shFrame.t;
		else if (thres < 0.5f + 2 * k)
			traceRay.d = -its.shFrame.t;
		else if (thres < 0.5f + 3 * k)
			traceRay.d = its.shFrame.s;
		else
			traceRay.d = -its.shFrame.s;

		//traceRay.d = -its.shFrame.n;

        //bRec.dx  = Warp::squareToExpDecayDisk(sigma_sample_dx, sampler.next2D());
        bRec.dx  = diffuseScatterPDFsample(sampler.next2D());
		//bRec.dx = 0;
        //cout << bRec.dx.toString() << endl;
        Vector3f disk3d(bRec.dx.x(), bRec.dx.y(), 0);
        Frame frame(-traceRay.d);
		traceRay.o = its.p + frame.toWorld(disk3d);

        
        float eps = 1e-3;
        traceRay.mint = - eps; //-bRec.dx.norm()-eps;
        traceRay.maxt = bRec.dx.norm() + eps;
        traceRay.update();
        //cout << bRec.dx.toString() << endl;

        bRec.probeRay = traceRay,
        bRec.probe = true;

        //bRec.dxPdf = diffuseScatterPDF(bRec.dx);
    }


    /// Draw a a sample from the BRDF model
    virtual Color3f sample(BSDFQueryRecord &bRec, const Intersection& its, const Point2f &sample, const Point2f &sample2) const override {

        float costhetai = Frame::cosTheta(bRec.wi);

		float extIOR = m_extIOR;
	   	float intIOR = m_intIOR;

        if (costhetai <= 0)
		{
			extIOR = m_intIOR;
			intIOR = m_extIOR;
		}

        bRec.measure = ESolidAngle;

        float eta = extIOR / intIOR; 
        bRec.eta = eta;

        // diffuse scattering
        bRec.isDelta = false;
        bRec.wo = Warp::squareToCosineHemisphere(sample2);

        if (bRec.measure != ESolidAngle
                || Frame::cosTheta(bRec.wi) <= 0
                || Frame::cosTheta(bRec.wo) <= 0)
            return 0.0f;

        float F = fresnelTransmission(bRec);

        // INV_PI*albedo*R(r)*costhetaoi/ (R(r)*INV_PI*costhetaoi*bRec.dxAxisPdf)
        return F*m_albedo->eval(bRec.uv); // bRec.dxAxisPdf;
    }

    bool isDiffuse() const {
        return true;
    }

    /// Return a human-readable summary
    virtual std::string toString() const override {
        return tfm::format(
            "Subsurface[\n"
            "  albedo = %s\n"
            "]",
            m_albedo ? indent(m_albedo->toString()) : std::string("null")
        );
    }

    virtual EClassType getClassType() const override { return EBSDF; }

private:

	// non derived
    Texture<Color3f> * m_albedo;
    float m_extIOR;
    float m_intIOR;

	Color3f sigma_t;

	// derived
    Color3f dmfp;
    Color3f s;
};

NORI_REGISTER_CLASS(Subsurface, "subsurface");
NORI_NAMESPACE_END
