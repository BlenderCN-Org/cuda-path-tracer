//
// Created by lukas on 18.12.17.
//

#ifndef NORIGL_ENVIROMENTLIGHT_H
#define NORIGL_ENVIROMENTLIGHT_H
#include <cmath>
#include <nori/emitter.h>
#include <nori/warp.h>
#include <nori/shape.h>
#include <nori/vector.h>
#include <iostream>
#include <nori/dpdf.h>
#include <nori/imagetexture.h>
#include <cuda_runtime_api.h>

NORI_NAMESPACE_BEGIN

    class EnviromentMap: public NoriObject{
    public:
        enum envType {ESphere} envType;
        __host__ virtual size_t getSize() const override {return sizeof(EnviromentMap);};
        __host__   virtual EClassType getClassType() const {return EEnvMap;};

        //__device__  Color3f eval(const EmitterQueryRecord & lRec) const ;
        //__device__  Color3f sample(EmitterQueryRecord & lRec, const Point2f & sample) const ;
        //__device__  float pdf(const EmitterQueryRecord &lRec) const  ;
    protected:

    };
NORI_NAMESPACE_END
#define CallMap(object,function,...) CallCuda1(EnviromentMap,envType,object,function,const,ESphere,SphereMap,__VA_ARGS__);
#endif //NORIGL_ENVIROMENTLIGHT_H
