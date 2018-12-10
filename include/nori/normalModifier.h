#if !defined(__NORI_NORMAL_MODIFIER_H)
#define __NORI_NORMAL_MODIFIER_H

#include <nori/object.h>

NORI_NAMESPACE_BEGIN

/**
 * \brief Abstract integrator (i.e. a rendering technique)
 *
 * In Nori, the different rendering techniques are collectively referred to as 
 * integrators, since they perform integration over a high-dimensional
 * space. Each integrator represents a specific approach for solving
 * the light transport equation---usually favored in certain scenarios, but
 * at the same time affected by its own set of intrinsic limitations.
 */
class NormalModifier : public NoriObject {
public:
    virtual ~NormalModifier() { }

    virtual void adjustNormal(Vector3f& normal, const Point2f& uv) const = 0;

    /**
     * \brief Return the type of object (i.e. Mesh/BSDF/etc.) 
     * provided by this instance
     * */
    virtual EClassType getClassType() const override { return ENormalModifier; }
};

NORI_NAMESPACE_END

#endif /* __NORI_NORMAL_MODIFIER_H */
