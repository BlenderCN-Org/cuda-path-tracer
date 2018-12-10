/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Wenzel Jakob

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

#include <nori/integrator.h>
#include <nori/sampler.h>
#include <nori/emitter.h>
#include <nori/bsdf.h>
#include <nori/scene.h>
#include <nori/photon.h>


NORI_NAMESPACE_BEGIN

class PhotonMapper : public Integrator {
public:
    /// Photon map data structure
    typedef PointKDTree<Photon> PhotonMap;

    PhotonMapper(const PropertyList &props) {
        /* Lookup parameters */
        m_photonCount  = props.getInteger("photonCount", 1000000);
        m_photonRadius = props.getFloat("photonRadius", 0.0f /* Default: automatic */);
    }

	int emitted_photons;

    virtual void preprocess(const Scene *scene) override {
        cout << "Gathering " << m_photonCount << " photons .. ";
        cout.flush();

        /* Create a sample generator for the preprocess step */
        Sampler *sampler = static_cast<Sampler *>(
            NoriObjectFactory::createInstance("independent", PropertyList()));

        /* Allocate memory for the photon map */
        m_photonMap = std::unique_ptr<PhotonMap>(new PhotonMap());
        m_photonMap->reserve(m_photonCount);

		/* Estimate a default photon radius */
		if (m_photonRadius == 0)
			m_photonRadius = scene->getBoundingBox().getExtents().norm() / 500.0f;

		int n_photons = 0;
		Ray3f ray;
		float successProb = 0.99f;
		
		std::vector<Emitter*> lights = scene->getLights();
		int nLights = lights.size();
		
		while(n_photons < m_photonCount)
		{
			Color3f W = scene->getRandomEmitter(sampler->next1D())->samplePhoton(ray, sampler->next2D(),sampler->next2D());
			Color3f Wprime = W;
			emitted_photons++;
			//bool emitted = false;
			

			//cout << W << endl;
			while(n_photons < m_photonCount)
			{
				// trace ray
				Intersection its;
				if (!scene->rayIntersect(ray, its)) break;

				const BSDF* bsdf = its.mesh->getBSDF();

				if (bsdf->isDiffuse())
				{
					 m_photonMap->push_back(Photon(
					 	its.p,  // Position
					 	-ray.d, // Direction
					 	W*nLights   // Power
					 ));
					n_photons++;
					//cout << W << endl;
				
					//emitted = true;
				}


				Vector3f wi = -ray.d;

				BSDFQueryRecord bsdfQuery(its.shFrame.toLocal(wi).normalized());
				bsdfQuery.uv = its.uv;
				Color3f f = bsdf->sample(bsdfQuery, its, sampler->next2D(), sampler->next2D());

				Vector3f light_dir = its.toWorld(bsdfQuery.wo).normalized();

				// set new wi
				ray.d = light_dir;
				ray.o = its.p;
				ray.mint = 1e-4;
				ray.update();

				Wprime = W*f;

				//successProb = std::min(1.0f, Wprime.maxCoeff()/W.maxCoeff());
				successProb = std::min(0.99f, W.maxCoeff());
				//cout << successProb << endl;
				//successProb = 0.999;
				if (sampler->next1D() > successProb)
				{
					sampler->advance();
					//cout << "kill" << endl;
					break;
				}
				else
				{
					Wprime /= successProb;
					sampler->advance();
					//cout << "bounce" << endl;
				}

				W = Wprime;
			}
		}

		/* How to add a photon?
		 * m_photonMap->push_back(Photon(
		 *	Point3f(0, 0, 0),  // Position
		 *	Vector3f(0, 0, 1), // Direction
		 *	Color3f(1, 2, 3)   // Power
		 * ));
		 */

		// put your code to trace photons here

		/* Build the photon map */
        m_photonMap->build();
    }

    virtual Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &_ray) const override {
    	
		/* How to find photons?
		 * std::vector<uint32_t> results;
		 * m_photonMap->search(Point3f(0, 0, 0), // lookup position
		 *                     m_photonRadius,   // search radius
		 *                     results);
		 *
		 * for (uint32_t i : results) {
		 *    const Photon &photon = (*m_photonMap)[i];
		 *    cout << "Found photon!" << endl;
		 *    cout << " Position  : " << photon.getPosition().toString() << endl;
		 *    cout << " Power     : " << photon.getPower().toString() << endl;
		 *    cout << " Direction : " << photon.getDirection().toString() << endl;
		 * }
		 */
		Ray3f traceRay = _ray;

		Color3f t = 1.0f;
		Color3f Li = 0.0f;
		float successProb = 0.99;

		while (true)
		{
			// trace ray
			Intersection its;
			if (!scene->rayIntersect(traceRay, its)) break;

			const BSDF* bsdf = its.mesh->getBSDF();

			if (its.mesh->isEmitter())
			{
				const Emitter* light = its.mesh->getEmitter();

				EmitterQueryRecord emitterQuery(traceRay.o, its.p, its.shFrame.n);
				Li += t*light->eval(emitterQuery);
			}

			if (bsdf->isDiffuse())
			{
				std::vector<uint32_t> results;
				m_photonMap->search(its.p, // lookup position
									  m_photonRadius,   // search radius
									  results);
				 
				Color3f val = 0;
				for (uint32_t i : results) {
					const Photon &photon = (*m_photonMap)[i];

					/*cout << "Found photon!" << endl;
					cout << " Position  : " << photon.getPosition().toString() << endl;
					cout << " Power     : " << photon.getPower().toString() << endl;
					cout << " Direction : " << photon.getDirection().toString() << endl;*/

					Vector3f cam_dir = -traceRay.d;
					Vector3f light_dir = photon.getDirection();
					BSDFQueryRecord bsdfQuery(its.shFrame.toLocal(cam_dir), its.shFrame.toLocal(light_dir), ESolidAngle);
					//BSDFQueryRecord bsdfQuery(its.shFrame.toLocal(photon.getDirection()).normalized());
					val += photon.getPower()*bsdf->eval(bsdfQuery);
					//cout << val << endl;
				}
				Li += t*val/(M_PI*m_photonRadius*m_photonRadius*emitted_photons);
				break;
			}

     		successProb = std::min(0.99f, t.maxCoeff());
			if (sampler->next1D() > successProb)
			{
				sampler->advance();
				break;
			}
			else
			{
				t /= successProb;
				sampler->advance();
			}

			//const BSDF* bsdf = its.mesh->getBSDF();
			Vector3f wi = -traceRay.d;

			BSDFQueryRecord bsdfQuery(its.toLocal(wi));
			bsdfQuery.uv = its.uv;
			Color3f f = bsdf->sample(bsdfQuery, its, sampler->next2D(), sampler->next2D());
			//cout << f.toString() << endl;

			Vector3f light_dir = its.toWorld(bsdfQuery.wo);

			// set new wi
			traceRay.d = light_dir.normalized();
			traceRay.o = its.p;
			traceRay.mint = 1e-4;
			traceRay.update();

			t *= f;
		}

		return Li;
    }

    virtual std::string toString() const override {
        return tfm::format(
            "PhotonMapper[\n"
            "  photonCount = %i,\n"
            "  photonRadius = %f\n"
            "]",
            m_photonCount,
            m_photonRadius
        );
    }
private:
    int m_photonCount;
    float m_photonRadius;
    std::unique_ptr<PhotonMap> m_photonMap;
};

NORI_REGISTER_CLASS(PhotonMapper, "photonmapper");
NORI_NAMESPACE_END
