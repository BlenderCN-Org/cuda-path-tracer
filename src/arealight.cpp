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
#include <nori/warp.h>
#include <nori/shape.h>
#include <nori/frame.h>
#include <iostream>

NORI_NAMESPACE_BEGIN

class AreaEmitter : public Emitter {
public:
    AreaEmitter(const PropertyList &props) {
        m_radiance = props.getColor("radiance");
    }

    virtual std::string toString() const override {
        return tfm::format(
                "AreaLight[\n"
                "  radiance = %s,\n"
                "]",
                m_radiance.toString());
    }

    virtual Color3f eval(const EmitterQueryRecord & lRec) const override {
        if(!m_shape)
            throw NoriException("There is no shape attached to this Area light!");

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

    virtual Color3f sample(EmitterQueryRecord & lRec, const Point2f & sample) const override {
        if(!m_shape)
            throw NoriException("There is no shape attached to this Area light!");

		ShapeQueryRecord shapeQuery(lRec.ref);
		m_shape->sampleSurface(shapeQuery, sample);

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

		if (fabs(pdfv) > 1e-6)
			return m_radiance/pdfv;
		else return Color3f(0.0f);
    }

    virtual float pdf(const EmitterQueryRecord &lRec) const override {
        if(!m_shape)
            throw NoriException("There is no shape attached to this Area light!");
		
		float dist2 = (lRec.ref - lRec.p).squaredNorm();
		float costheta = -lRec.wi.normalized().dot(lRec.n.normalized()); 

		if (costheta <= 0)
		{
			return 0.0f;
		}
		else
		{
			ShapeQueryRecord shapeQuery(lRec.ref, lRec.p);
			return m_shape->pdfSurface(shapeQuery)*dist2/fabs(costheta);
		}
    }


    virtual Color3f samplePhoton(Ray3f &ray, const Point2f &sample1, const Point2f &sample2) const override {

        if(!m_shape)
            throw NoriException("There is no shape attached to this Area light!");

		ShapeQueryRecord shapeQuery(Vector3f({}));
		m_shape->sampleSurface(shapeQuery, sample1);

        Vector3f in_dir = Warp::squareToCosineHemisphere(sample2);
		//Vector3f ref = shapeQuery.p + in_dir;

		Frame frame(shapeQuery.n);

		in_dir = frame.toWorld(in_dir);
		Vector3f ref = shapeQuery.p + in_dir;

		// p - ref = -in_dir
		ray.d = in_dir.normalized();
		ray.o = shapeQuery.p;
		ray.mint = 1e-4;
		ray.update();

		EmitterQueryRecord emitterQuery(ref, shapeQuery.p, shapeQuery.n);

		float A = 1.0f/shapeQuery.pdf;
		//Color3f Le = eval(emitterQuery);
		Color3f Le = m_radiance;

		//cout << Le << endl;

		return M_PI*A*Le;
    }


protected:
    Color3f m_radiance;
};

NORI_REGISTER_CLASS(AreaEmitter, "area")
NORI_NAMESPACE_END
