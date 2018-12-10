#if !defined(_NORIGL_NORMAL_MODIFIER_H)
#define _NORIGL_NORMAL_MODIFIER_H

#include <nori/object.h>
#include <nori/frame.h>
#include <nori/bbox.h>

NORI_NAMESPACE_BEGIN

class NormalModifier : public NoriObject {
public:

    NormalModifier() {};
    virtual ~NormalModifier() {};

    enum modifierType {EBumpMap,ENormalMap} modifierType;

    virtual size_t getSize() const  override  {return sizeof(NormalModifier);}

    /**
     * \brief Return the type of object (i.e. Mesh/BSDF/etc.)
     * provided by this instance
     * */
    virtual EClassType getClassType() const override { return ENormalModifier; }


};
//#define CallNormalModifier(object,function,...) CallCuda1(NormalModifier,modifierType,object,function,nonconst,EBumpMap,BumpMap,__VA_ARGS__)
#define CallNormalModifier(object,function,...) CallCuda2(NormalModifier,modifierType,object,function,nonconst,EBumpMap,BumpMap, ENormalMap,NormalMap, __VA_ARGS__)

NORI_NAMESPACE_END

#endif /* _NORIGL_NORMAL_MODIFIER_H */
