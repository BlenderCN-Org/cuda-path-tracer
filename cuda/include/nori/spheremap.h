//
// Created by lukas on 18.12.17.
//

#ifndef NORIGL_SPHEREMAP_H
#define NORIGL_SPHEREMAP_H
#include <nori/enviromentmap.h>
#include <nori/frame.h>

NORI_NAMESPACE_BEGIN
class SphereMap: public EnviromentMap{
public:
    __host__ virtual size_t getSize() const override {return sizeof(SphereMap);};
    __host__ SphereMap(const PropertyList &props);
    __host__ virtual std::string toString() const override ;
    __host__ virtual void gpuTransfer(NoriObject **o) override ;
    __host__ virtual void activate() override;
    __device__  Color3f eval(const EmitterQueryRecord & lRec) const ;
    __device__  Color3f sample(EmitterQueryRecord & lRec, const Point2f & sample) const ;
    __device__  float pdf(const EmitterQueryRecord &lRec) const  ;
protected:
    ImageTexture<Color3f>* m_radiance;
    DiscretePDF m_pdf;
    size_t w;
    size_t h;
    float m_bright;
    Frame origin;
};

NORI_NAMESPACE_END
#endif //NORIGL_SPHEREMAP_H
