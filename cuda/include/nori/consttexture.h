//
// Created by lukas on 25.11.17.
//

#ifndef NORIGL_CONSTTEXTURE_H
#define NORIGL_CONSTTEXTURE_H

#include <nori/texture.h>

NORI_NAMESPACE_BEGIN
template <typename T>
class ConstantTexture : public Texture<T> {
public:
    virtual size_t getSize() const override {return sizeof(ConstantTexture<T>);};
    ConstantTexture(const PropertyList &props);
    virtual std::string toString() const override;
    __device__ T eval(const Point2f & uv){return m_value;};
protected:
    T m_value;
};


NORI_NAMESPACE_END
#endif //NORIGL_CONSTTEXTURE_H
