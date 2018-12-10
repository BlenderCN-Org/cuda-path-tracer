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
#include <nori/consttexture.h>
#include <nori/imagetexture.h>
#include <nori/independent.h>
#include <nori/subsurface.h>

NORI_NAMESPACE_BEGIN

/**
 * \brief Subsurface / Lambertian BRDF model
 */

#ifndef __CUDA_ARCH__

__host__ Subsurface::Subsurface(const PropertyList &propList) : m_albedo(nullptr) {
    if(propList.has("albedo")) {
        PropertyList l;
        l.setColor("value", propList.getColor("albedo"));
        m_albedo = static_cast<Texture<Color3f> *>(NoriObjectFactory::createInstance("constant_color", l));
    }

    m_intIOR = propList.getFloat("intIOR", 1.3046f);
    m_extIOR = propList.getFloat("extIOR", 1.000277f);

    dmfp = propList.getColor("dmfp", 0.1f);

    isDiffuse = true;
    this->bsdfType = TSubsurface;
}

__host__ Subsurface::~Subsurface() {
    cudaFree(m_albedo);
}


/// Add texture for the albedo
__host__ void Subsurface::addChild(NoriObject *obj) {
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

__host__ void Subsurface::activate() {
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

/// Return a human-readable summary
__host__ std::string Subsurface::toString() const {
    return tfm::format(
        "Subsurface[\n"
        "  albedo = %s\n"
        "]",
        m_albedo ? indent(m_albedo->toString()) : std::string("null")
    );
}

__host__ void Subsurface::gpuTransfer(NoriObject ** objects) {
    this->m_albedo = (Texture <Color3f> *) objects[m_albedo->objectId];
    //do not delete object now
    //delete tmp;
}

#endif

__device__ float Subsurface::colorNorm(const Color3f& c) const
{
    return sqrt(c.x()*c.x() + c.y()*c.y() + c.z()*c.z());
}

/// Evaluate the BRDF model
__device__ Color3f Subsurface::eval(const BSDFQueryRecord &bRec) const {

    //we want: return diffuseScatteringApprox() / bRec.dxPdf(); -> R(r) cancels and this gives:
    float F = fresnelTransmission(bRec);
    return F*CallTexture(m_albedo, Color3f, eval, bRec.uv)*INV_PI;
}

/*__device__ Vector3f Subsurface::expVec(const Vector3f &v) const {

    return Vector3f(exp(v.x()), exp(v.y()), exp(v.z()));
}*/

__device__ float Subsurface::diffuseScatterApproxCDF(float r) const
{
    float rsd = -r*colorNorm(s)/colorNorm(dmfp);
    return 1.0f - 0.25f*std::exp(rsd)-0.75f*std::exp(rsd/3.0f);
}

__device__ float Subsurface::diffuseScatterApproxCDFDerivative(float r) const
{
    float sd = colorNorm(s)/colorNorm(dmfp);
    float rsd = -r*sd;
    return sd * 0.25f* (std::exp(rsd) + std::exp(rsd/3.0f));
}

__device__ Point2f Subsurface::diffuseScatterPDFsample(const Point2f& sample2) const
{
    float sample = sample2.x();

    // newton find inverse of CDF
    float sdInv = colorNorm(dmfp)/colorNorm(s);
    float r0 = -log((1.0f - sample) * 4.0f/3.0f)*3.0f*sdInv;
    float r1 = -log((1.0f - sample) * 4.0f)*3.0f*sdInv;
    
    float err0 = fabs(diffuseScatterApproxCDF(r0) - sample);;
    float err1 = fabs(diffuseScatterApproxCDF(r1) - sample);

    if (err0 < err1)  r1=r0;
    else r1 = r0;

    //(cout << "|phi - cdf(r)|_0a=" << fabs(diffuseScatterApproxCDF(r0) - sample) << endl;
    //cout << "|phi - cdf(r)|_0b=" << fabs(diffuseScatterApproxCDF(r1) - sample) << endl;

    for (int i=0; i<2; ++i)
    {
        float f = (diffuseScatterApproxCDF(r0) - sample);
        float df = diffuseScatterApproxCDFDerivative(r0);
        r1 = r0 - f/df;
        r0 = r1;
    }

    //cout << "|phi - cdf(r)|_1=" << fabs(diffuseScatterApproxCDF(r0) - sample) << endl;
    
    float theta = 2.0f*M_PI*sample2.y();
    return Point2f(r0*cos(theta), r0*sin(theta));
}

__device__ float Subsurface::diffuseScatterPDF(const Point2f& p) const
{
    float sn = colorNorm(s);
    float dmfpn = colorNorm(dmfp);
    float r = p.norm();

    return sn*(std::exp(-sn*r/dmfpn) + std::exp(-sn*r/(3.0f*dmfpn)))/(8.0f*M_PI*dmfpn*r);
    //return diffuseScatterApproxCDFDerivative(r) / (2*M_PI*r);
}

__device__ Color3f Subsurface::diffuseScatteringApprox(const BSDFQueryRecord &bRec) const
{
    // approx (https://graphics.pixar.com/library/ApproxBSSRDF/paper.pdf equation 3)
    if (bRec.measure != ESolidAngle
        || Frame::cosTheta(bRec.wi) <= 0
        || Frame::cosTheta(bRec.wo) <= 0)
        return 0.0f;

    float r = bRec.probeDistance;

    Color3f albedo = CallTexture(m_albedo, Color3f, eval, bRec.uv);
    Color3f Rr = albedo*s*((-s*r/dmfp).exp() + (-s*r/(3.0f*dmfp)).exp())/(8.0f*M_PI*dmfp*r);
    float F = fresnelTransmission(bRec);
    return F*Rr*INV_PI;
}

__device__ float Subsurface::fresnelTransmission(const BSDFQueryRecord &bRec) const
{
    float Ft_i = 1.0f-fresnel(Frame::cosTheta(bRec.wi), m_extIOR, m_intIOR);
    float Ft_o = 1.0f-fresnel(Frame::cosTheta(bRec.wo), m_extIOR, m_intIOR);
    return Ft_i*Ft_o;
}

/// Compute the density of \ref sample() wrt. solid angles
__device__ float Subsurface::pdf(const BSDFQueryRecord &bRec) const {
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

    return INV_PI * Frame::cosTheta(bRec.wo);
}

__device__ void Subsurface::sampleDx(BSDFQueryRecord &bRec, const Intersection& its, Sampler& sampler) const
{
    // scatter ray test
    Ray3f traceRay;// = bRec.probeRay;

    float thres = CallSampler(sampler, next1D);
    float k = 0.5f/4.0f;

    if (thres < 0.5f)
        traceRay.d = -its.shFrame.n; // to surface parallel to normal
    else if (thres < 0.5f+k)
        traceRay.d = its.shFrame.t; 
    else if (thres < 0.5f+2*k)
        traceRay.d = -its.shFrame.t; 
    else if (thres < 0.5f+3*k)
        traceRay.d = its.shFrame.s; 
    else 
        traceRay.d = -its.shFrame.s; 
    

    Point2f dx  = diffuseScatterPDFsample(CallSampler(sampler,next2D));
    //cout << dx.toString() << endl;
    Vector3f disk3d(dx.x(), dx.y(), 0);
    Frame frame(-traceRay.d);
    traceRay.o = its.p + frame.toWorld(disk3d);
    
    float eps = 1e-2;
    traceRay.mint = -eps;
    traceRay.maxt = dx.norm()+eps;
    traceRay.update();

    bRec.probeRay = traceRay;
    bRec.probe = true;
}


/// Draw a a sample from the BRDF model
__device__ Color3f Subsurface::sample(BSDFQueryRecord &bRec, const Point2f &sample) const
{
    if (Frame::cosTheta(bRec.wi) <= 0)
        return 0.0f;

    float costhetai = Frame::cosTheta(bRec.wi);

    float eta = m_extIOR / m_intIOR; 
    bRec.eta = eta;

    // diffuse scattering
    bRec.isDelta = false;
    bRec.wo = Warp::squareToCosineHemisphere(sample);

    float F = fresnelTransmission(bRec);

    // we want:
    // INV_PI*albedo*R(r)*costhetaoi/ (R(r)*INV_PI*costhetaoi*bRec.dxAxisPdf)
    // but almost all terms cancels except:
    return F*CallTexture(m_albedo, Color3f, eval, bRec.uv); // bRec.dxAxisPdf;
}


#ifndef __CUDA_ARCH__
    NORI_REGISTER_CLASS(Subsurface, "subsurface");
#endif
NORI_NAMESPACE_END
