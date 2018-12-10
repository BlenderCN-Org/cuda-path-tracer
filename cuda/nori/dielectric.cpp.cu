#include <nori/dielectric.h>
#include <nori/bsdf.h>
#include <nori/frame.h>
#include <nori/common.h>

NORI_NAMESPACE_BEGIN

/// Ideal dielectric BSDF

__host__ Dielectric::Dielectric(const PropertyList &propList) {
    /* Interior IOR (default: BK7 borosilicate optical glass) */
    m_intIOR = propList.getFloat("intIOR", 1.5046f);

    /* Exterior IOR (default: air) */
    m_extIOR = propList.getFloat("extIOR", 1.000277f);
    
    this->bsdfType = TDielectric;
}

__host__ std::string Dielectric::toString() const 
{
    return tfm::format(
        "Dielectric[\n"
        "  intIOR = %f,\n"
        "  extIOR = %f\n"
        "]",
        m_intIOR, m_extIOR);
}

__device__ Color3f Dielectric::eval(const BSDFQueryRecord &) const  {
    /* Discrete BRDFs always evaluate to zero in Nori */
    return Color3f(0.0f);
}

__device__ float Dielectric::pdf(const BSDFQueryRecord &) const  {
    /* Discrete BRDFs always evaluate to zero in Nori */
    return 0.0f;
}

__device__ Color3f Dielectric::sample(BSDFQueryRecord &bRec, const Point2f &sample) const {
    
    float costhetai = Frame::cosTheta(bRec.wi);

    float extIOR = m_extIOR;
    float intIOR = m_intIOR;
    if (costhetai < 0)
    {
        extIOR = m_intIOR;
        intIOR = m_extIOR;
        //return Color3f(0.0f);
    }
    float fresnel_val = fresnel(fabsf(costhetai), extIOR, intIOR);
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
        bRec.wo = -IORrel*(bRec.wi - win*n) - n*sqrtf(1.0f - IORrel*IORrel*(1.0f - win*win));
        bRec.wo.normalize();

        bRec.measure = EDiscrete;
        bRec.eta = IORrel;

        float IORrelInv = 1.0f/IORrel;
        return Color3f(IORrelInv*IORrelInv);
        //return Color3f(1.0f);
    }

    return Color3f(1.0f);
}

__host__ void Dielectric::gpuTransfer(NoriObject ** objects) {
    //this->m_albedo = (Texture <Color3f> *) objects[m_albedo->objectId];
}



#ifndef __CUDA_ARCH__
    NORI_REGISTER_CLASS(Dielectric, "dielectric");
#endif
NORI_NAMESPACE_END
