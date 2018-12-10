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

#include <nori/shape.h>
#include <nori/bsdf.h>
#include <nori/emitter.h>
#include <nori/warp.h>

#include <Eigen/Geometry>

NORI_NAMESPACE_BEGIN

class Sphere : public Shape {
public:
    Sphere(const PropertyList & propList) {
        m_position = propList.getPoint3("center", Point3f());
        m_radius = propList.getFloat("radius", 1.f);

        m_bbox.expandBy(m_position - Vector3f(m_radius));
        m_bbox.expandBy(m_position + Vector3f(m_radius));
    }

    virtual BoundingBox3f getBoundingBox(uint32_t index) const override { return m_bbox; }

    virtual Point3f getCentroid(uint32_t index) const override { return m_position; }

    virtual bool rayIntersect(uint32_t index, const Ray3f &ray, float &u, float &v, float &t) const override {

		//float A = ray.d.transpose()*ray.d;
		// we assume A=1
		
		float B = 2*(ray.o - m_position).transpose()*ray.d;
		float C = -m_radius*m_radius + m_position.transpose()*m_position + ray.o.transpose()*ray.o - 2*m_position.transpose()*ray.o;

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
				locT = std::min(locT1, locT2);
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
			//if (locT2 >= 0) locT = std::min(locT, locT2);
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

    virtual void setHitInformation(uint32_t index, const Ray3f &ray, Intersection & its) const override {

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

		float theta = acos(frac) / (M_PI);
		float phi = (atan2(p_loc.y(),p_loc.x()) + M_PI) / (2*M_PI);

		its.uv = Point2f(phi, theta);
    }

    virtual void sampleSurface(ShapeQueryRecord & sRec, const Point2f & sample) const override {
        Vector3f q = Warp::squareToUniformSphere(sample);
        sRec.p = m_position + m_radius * q;
        sRec.n = q;
        //sRec.pdf = std::pow(1.f/m_radius,2) * Warp::squareToUniformSpherePdf(Vector3f(0.0f,0.0f,1.0f));
        sRec.pdf = 1.0/(4.0*M_PI*m_radius*m_radius); //* Warp::squareToUniformSpherePdf(Vector3f(0.0f,0.0f,1.0f));
    }
    virtual float pdfSurface(const ShapeQueryRecord & sRec) const override {
        return std::pow(1.f/m_radius,2) * Warp::squareToUniformSpherePdf(Vector3f(0.0f,0.0f,1.0f));
    }


    virtual std::string toString() const override {
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

protected:
    Point3f m_position;
    float m_radius;
};

NORI_REGISTER_CLASS(Sphere, "sphere");
NORI_NAMESPACE_END
