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

#include <nori/sphere.h>
#include <nori/shape.h>
#include <nori/bsdf.h>
#include <nori/emitter.h>
#include <nori/warp.h>
#include <nori/normalModifier.h>
#include <nori/bumpMap.h>

#include <Eigen/Geometry>

NORI_NAMESPACE_BEGIN


	__host__ Sphere::Sphere(const PropertyList & propList) {
		shapeType= ETSphere;
        m_position = propList.getPoint3("center", Point3f());
        m_radius = propList.getFloat("radius", 1.f);

        m_bbox.expandBy(m_position - Vector3f(m_radius));
        m_bbox.expandBy(m_position + Vector3f(m_radius));
    }

    __host__ BoundingBox3f Sphere::getBoundingBox(uint32_t index) const { return m_bbox; }

	__host__ __device__ Point3f Sphere::getCentroid(uint32_t index) const { return m_position; }

    __device__ bool Sphere::rayIntersect(uint32_t index, const Ray3f &ray, float &u, float &v, float &t) const {

		//float A = ray.d.transpose()*ray.d;
		// we assume A=1
		
		float B = 2*(ray.o - m_position).dot(ray.d);
		float C = -m_radius*m_radius + m_position.dot(m_position) + ray.o.dot(ray.o) - 2*m_position.dot(ray.o);

		//float discr = B*B - 4*A*C;
		float discr = B*B - 4*C;


		if (discr > 0)
		{
			// two solutions
			//float locT1 = (-B+sqrt(discr))/(2*A);
			//float locT2 = (-B-sqrt(discr))/(2*A);
			float locT1 = (-B+sqrt(discr))/(2.0);
			float locT2 = (-B-sqrt(discr))/(2.0);
			float locT = locT2;
			
			bool t1Valid = locT1 >= ray.mint && locT1 <= ray.maxt;
			bool t2Valid = locT2 >= ray.mint && locT2 <= ray.maxt;

			if (t1Valid && t2Valid)
			{
				locT = stdmin(locT1, locT2);
			}
			else if (t1Valid)
			{
				locT = locT1;
			}
			else if (t2Valid)
			{
				locT = locT2;
			}
			else
			{
				return false;
			}

			//if (locT2 >= 0) locT = locT2;
			//if (locT2 >= 0) locT = stdmin(locT, locT2);
			//else std::cout << "err" << endl;
			
			//if (locT >= ray.mint && locT <= ray.maxt)
			{
				t = locT;
				return true;
			}
		}
		else if (discr == 0)
		{
			// single solution
			float locT = -B/(2.0);

			if (locT >= ray.mint && locT < ray.maxt)
			{
				t = locT;
				return true;
			}
		}
		else
		{
			// no solution
			return false;
		}


        return false;

    }

	__device__  void Sphere::setHitInformation(uint32_t index, const Ray3f &ray, Intersection & its) const  {

		its.p = ray(its.t); // eval ray at time t
		its.shFrame = Frame((its.p - m_position).normalized());

		if ((ray(ray.mint) - m_position).squaredNorm() <= m_radius*m_radius)
		{
			//its.shFrame = Frame((-its.p + m_position).normalized());
		}

		its.geoFrame = its.shFrame;

		Point3f p_loc = its.p - m_position;

        
        // fix slightly out of boundary values
        float frac = p_loc.z()/m_radius;
        if (frac < -1)
            frac = -1;
        else if (frac > 1)
            frac = 1;

		float theta = acosf(frac) / (M_PI);
		float phi = (atan2f(p_loc.y(),p_loc.x()) + M_PI) / (2*M_PI);

		its.uv = Point2f(phi, theta);

        if (m_normalModifier != NULL)
        {
            Frame f = its.shFrame;
            its.shFrame = Frame(CallNormalModifier(m_normalModifier, eval, its.uv, its.shFrame));
        }
    }

	__device__ void Sphere::sampleSurface(ShapeQueryRecord & sRec, const Point2f & sample) const  {
        Vector3f q = Warp::squareToUniformSphere(sample);
        sRec.p = m_position + m_radius * q;
        sRec.n = q;
        //sRec.pdf = std::pow(1.f/m_radius,2) * Warp::squareToUniformSpherePdf(Vector3f(0.0f,0.0f,1.0f));
        sRec.pdf = 1.0/(4.0*M_PI*m_radius*m_radius); //* Warp::squareToUniformSpherePdf(Vector3f(0.0f,0.0f,1.0f));
    }
	__device__ float Sphere::pdfSurface(const ShapeQueryRecord & sRec)  const {
        return powf(1.f/m_radius,2) * Warp::squareToUniformSpherePdf(Vector3f(0.0f,0.0f,1.0f));
    }


    __host__  std::string Sphere::toString() const  {
        return tfm::format(
                "Sphere[\n"
                "  center = %s,\n"
                "  radius = %f,\n"
                "  bsdf = %s,\n"
                "  emitter = %s\n"
                "]",
                m_position.toString(),
                m_radius,
                m_bsdf ? indent(m_bsdf->toString()) : std::string("null"),
                m_emitter ? indent(m_emitter->toString()) : std::string("null"));
    }


#ifndef __CUDA_ARCH__
	NORI_REGISTER_CLASS(Sphere, "sphere");
#endif
NORI_NAMESPACE_END
