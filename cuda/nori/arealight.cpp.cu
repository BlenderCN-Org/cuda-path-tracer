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
#include <nori/arealight.h>
#include <nori/mesh.h>
#include <nori/shape.h>
#include <nori/frame.h>
#include <iostream>
#include <cuda_runtime_api.h>

NORI_NAMESPACE_BEGIN

   __host__ AreaEmitter::AreaEmitter(const PropertyList &props){
		this->emitterType = EArea;
        m_radiance = props.getColor("radiance");
   }

   __host__ std::string AreaEmitter::toString() const  {
        return tfm::format(
                "AreaLight[\n"
                "  radiance = %s,\n"
                "]",
                m_radiance.toString());
    }

     __device__ Color3f AreaEmitter::eval(const EmitterQueryRecord & lRec) const {
        //printf("Printf Debugging: %p %f\n",this,m_radiance(0));
        if(!m_shape)
            return 0;
            // throw NoriException("There is no shape attached to this Area light!");
        //printf("%f\n",m_radiance.x());
		// wi points into the light source, n out of the lightsource
		if (lRec.n.dot(lRec.wi) < 0)
		{
			return m_radiance;
		}
		else
		{
			return 0;
		}
    }

    __device__ Color3f AreaEmitter::sample(EmitterQueryRecord & lRec, const Point2f & sample) const {
        if(!m_shape)
            return 0;
            //throw NoriException("There is no shape attached to this Area light!");

		ShapeQueryRecord shapeQuery(lRec.ref);
	    CallShape(m_shape,sampleSurface,shapeQuery, sample);
		//std::cout << shapeQuery.p << std::endl;
		//std::cout << std::endl;

		// points from ref to light emission point
		Vector3f dir = (shapeQuery.p - lRec.ref);

		float dist = dir.norm();
		lRec.wi = dir.normalized();
		lRec.n = shapeQuery.n;
		lRec.p = shapeQuery.p;

        //printf("ref=%.2f,p=%.2f \n", lRec.ref.norm(), shapeQuery.p.norm());

		// ray from light emission point to ref (object)
		lRec.shadowRay.o = shapeQuery.p;
		lRec.shadowRay.d = -lRec.wi;
		float delta = 1e-3;
		lRec.shadowRay.mint = delta;
		lRec.shadowRay.maxt = dist-delta;
		lRec.shadowRay.update();


		//std::cout << (lRec.shadowRay(0) - shapeQuery.p) << std::endl << std::endl;
		//std::cout << dist << std::endl;
		//
		//float pdfv = pdf(lRec);

		/*float dist2 = (shapeQuery.p - lRec.ref).squaredNorm();
		float costheta = -lRec.wi.normalized().dot(lRec.n.normalized());
		float pdfv = shapeQuery.pdf*dist2/costheta;*/
		float pdfv = pdf(lRec);
		lRec.pdf = pdfv;

		//float costheta = -lRec.wi.normalized().dot(lRec.n.normalized());

		if (fabs(pdfv) > float(1e-6))
			return m_radiance/pdfv;
		else return Color3f(0.0f);
    }

   __device__ float AreaEmitter::pdf(const EmitterQueryRecord &lRec) const {
        if(!m_shape)
            return 0;
            // throw NoriException("There is no shape attached to this Area light!");

		float dist2 = (lRec.ref - lRec.p).squaredNorm();
		float costheta = -lRec.wi.normalized().dot(lRec.n.normalized());
		if (costheta <= 0)
		{
			return 0.0f;
		}
		else
		{
			ShapeQueryRecord shapeQuery(lRec.ref, lRec.p);

			return CallShape(m_shape,pdfSurface,shapeQuery)*dist2/fabs(costheta);
		}
    }


#ifndef __CUDA_ARCH__
    NORI_REGISTER_CLASS(AreaEmitter, "area")
#endif
NORI_NAMESPACE_END
