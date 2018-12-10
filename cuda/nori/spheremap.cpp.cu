
/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Romain Prévost

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
#include <cmath>
#include <nori/emitter.h>
#include <nori/sphere.h>
#include <nori/warp.h>
#include <nori/spheremap.h>
#include <nori/mesh.h>
#include <nori/shape.h>
#include <nori/frame.h>
#include <iostream>
#include <cuda_runtime_api.h>

NORI_NAMESPACE_BEGIN
#ifndef __CUDA_ARCH__
    __host__ SphereMap::SphereMap(const PropertyList &props){
        //load texture
        this->m_bright = props.getFloat("brightness_factor",1);
        Vector3f v = props.getVector3("origin",Vector3f(0,1,0));
        origin = Frame(v);

        this->m_radiance = static_cast<
                ImageTexture<
                Color3f> *>(
                NoriObjectFactory::createInstance("image_color", props));
    }

    __host__ std::string SphereMap::toString() const  {
        return tfm::format(
                "EnviromentLight[\n"
                        "  texture = %s,\n"
                        "]",
                m_radiance->toString());
    }
    __host__ void SphereMap::gpuTransfer(NoriObject **o)  {
        this->m_radiance = (ImageTexture<
                            Color3f> *)o[m_radiance->objectId];
        m_pdf.transferGpu();
    }

    __host__ void SphereMap::activate() {
        cv::Mat m = m_radiance->getImage();
        this->w = m.rows;
        this->h = m.cols;
        m_pdf.reserve(w * h);
        for (int i = 0; i < w; i++){
            for (int j = 0; j < h; j++) {
                auto v  = m.at<cv::Vec3b>(cv::Point(j , i));
                float va = float(v(1)+v(0)+v(2))/255;
                va *= sin(M_PI*float(i)/h);
                m_pdf.append(va);
            }
        }
        m_pdf.normalize();
    }

#endif

__device__ Color3f SphereMap::eval(const EmitterQueryRecord & lRec) const {
        Vector3f p = lRec.wi;
        
        p = origin.toLocal(p);

        float u = acosf(p.z()) / (M_PI);
        float v = (atan2f(p.y(),p.x()) + M_PI) / (2*M_PI);
        //printf("u %f ,v %f",u,v);
        return m_radiance->eval(Point2f(u,v))*m_bright;
    }

__device__ Color3f SphereMap::sample(EmitterQueryRecord & lRec, const Point2f & sample) const {
    //use underlying discrete pdf to sample
    size_t i = m_pdf.sample(sample(0));
    size_t v = i/h;
    size_t u = i%h;

    //
    Point2f uv(float(u)/w,float(v)/h);
    Point3f p(cosf(uv.x()*M_PI)*sinf(uv.y()*2*M_PI),sinf(uv.x()*M_PI)*sinf(uv.y()*2*M_PI),cosf(uv.x()*M_PI));
    p = origin.toWorld(p);
    lRec.p = p;
    lRec.wi = p;
    lRec.shadowRay = Ray3f(lRec.ref,lRec.wi,Epsilon,INFINITY);
    lRec.pdf = pdf(lRec);
    return m_radiance->eval(uv)*m_bright;
}

__device__ float SphereMap::pdf(const EmitterQueryRecord &lRec) const {
       /**todo**/
    Vector3f p = lRec.wi;
    p = origin.toLocal(p);
    
    float u = acosf(p.z()) / (M_PI);
    float v = (atan2f(p.y(),p.x()) + M_PI) / (2*M_PI);
    auto c = m_radiance->eval(Point2f(u,v));
    return (c(0)+c(1)+c(2))*(m_pdf.getNormalization())*sinf(u*M_PI);
}


#ifndef __CUDA_ARCH__
NORI_REGISTER_CLASS(SphereMap, "sphereMap");
#endif
NORI_NAMESPACE_END
