#include <nori/integrator.h>
#include <nori/scene.h>
#include <nori/warp.h>
#include <nori/bsdf.h>
#include <nori/vector.h>
#include <iostream>
#include <cmath>

NORI_NAMESPACE_BEGIN

class DirectMisIntegrator : public Integrator {
public:

	float maxLen;

    DirectMisIntegrator(const PropertyList &props) {

    }

    Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray) const {

        /* Find the surface that is visible in the requested direction */
        Intersection its;
        if (!scene->rayIntersect(ray, its))
            return Color3f(0.0f);

		std::vector<Emitter*> lights = scene->getLights();

		// initialize return color
		Color3f li(0.0f);

		// copy frame and uv coordinates
		Frame shFrame = its.shFrame;
		Point2f uv = its.uv;

		// we directly hit a light source
		if (its.mesh->getEmitter() != NULL)
		{
			const Emitter* light = its.mesh->getEmitter();

			EmitterQueryRecord emitterQuery(ray.o, its.p, its.shFrame.n);
			li += light->eval(emitterQuery);
		}

		const BSDF* bsdf = its.mesh->getBSDF();
		Vector3f cam_dir = -ray.d;


		// we indirectly might hit a light source
		// EMS PART	
		{
			for (Emitter* light : lights)
			{
				const BSDF* bsdf = its.mesh->getBSDF();

				// prepare light source and create shadowRay 
				EmitterQueryRecord emitterQuery(its.p);
				Color3f liLoc = light->sample(emitterQuery, sampler->next2D());

				Vector3f light_dir = emitterQuery.wi;
				Vector3f cam_dir = -ray.d;

				// check if the light actually hits the intersection point
				if (scene->rayIntersect(emitterQuery.shadowRay))
				{
					continue;
				}

				//BSDFQueryRecord bsdfquery2(shFrame.toLocal(light_dir), shFrame.toLocal(cam_dir), ESolidAngle);
				BSDFQueryRecord bsdfquery2(shFrame.toLocal(cam_dir), shFrame.toLocal(light_dir), ESolidAngle);
				bsdfquery2.uv = uv;
				

				// bsdf based color
				Color3f f = bsdf->eval(bsdfquery2);

				float bsdf_pdf = bsdf->pdf(bsdfquery2);
				float light_pdf = light->pdf(emitterQuery);
				float wem = light_pdf / (light_pdf + bsdf_pdf);

				// angle between light ray and surface normal
				float costhetaSurface = light_dir.normalized().dot(shFrame.n.normalized());

				//cout << costhetaSurface << endl;
				if (wem > 0)
				{
					li += wem*liLoc*f*costhetaSurface;
				}
			}
		}


		// MAT PART
		//

		{
			BSDFQueryRecord bsdfQuery(shFrame.toLocal(cam_dir));
			bsdfQuery.uv = uv;

			sampler->advance();
			Color3f f = bsdf->sample(bsdfQuery, its, sampler->next2D(), sampler->next2D());

			Vector3f light_dir_local = bsdfQuery.wo;
			Vector3f light_dir = its.toWorld(bsdfQuery.wo);

			Ray3f refl(its.p, light_dir);
			refl.mint = 1e-4;

			if (scene->rayIntersect(refl, its))
			{
				if (its.mesh->getEmitter() != NULL)
				{
					const Emitter* light = its.mesh->getEmitter();

					//Color3f f = bsdf->eval(bsdfQuery);
					
					if (refl.o - its.p == Vector3f(0))
					{
						cout << "err " << endl;
					}
					EmitterQueryRecord emitterQuery(refl.o, its.p, its.shFrame.n);
					Color3f flight = light->eval(emitterQuery);

					//BSDFQueryRecord bsdfQuery(shFrame.toLocal(light_dir), shFrame.toLocal(cam_dir), ESolidAngle);
					float bsdf_pdf = bsdf->pdf(bsdfQuery);
					float light_pdf = light->pdf(emitterQuery);
					float wmat = bsdf_pdf / (light_pdf + bsdf_pdf);

					if (wmat > 0)
					{
						li += wmat*f*flight;
					}
				}
			}
		}



		return li;
    }

    std::string toString() const {
        return "DirectMisIntegrator[]";
    }
};

NORI_REGISTER_CLASS(DirectMisIntegrator, "direct_mis");
NORI_NAMESPACE_END
