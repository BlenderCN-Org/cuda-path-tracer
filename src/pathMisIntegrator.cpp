#include <nori/integrator.h>
#include <nori/scene.h>
#include <nori/warp.h>
#include <nori/bsdf.h>
#include <iostream>

NORI_NAMESPACE_BEGIN

class PathMisIntegrator : public Integrator {
public:

	float maxLen;

    PathMisIntegrator(const PropertyList &props) {

    }

    Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &_ray) const {

        /* Find the surface that is visible in the requested direction */

		Ray3f traceRay = _ray;
		std::vector<Emitter*> lights = scene->getLights();

		// initialize return color
		Color3f li(0.0f);
		Color3f t = Color3f(1.0f);
		float successProb = 0.99;
		//const BSDF* last_bsdf = NULL;

        Intersection its;

		bool lastBsdfDelta = true;
		float last_bsdf_pdf = 1.0f;
		
		while (true)
		{
			// trace ray
			if (!scene->rayIntersect(traceRay, its)) break;

			// we directly hit a light source
			if (its.mesh->getEmitter() != NULL)
			{
				const Emitter* light = its.mesh->getEmitter();

				EmitterQueryRecord emitterQuery(traceRay.o, its.p, its.shFrame.n);
				Color3f flight = light->eval(emitterQuery);

				//BSDFQueryRecord bsdfQuery(shFrame.toLocal(light_dir), its.shFrame.toLocal(cam_dir), ESolidAngle);
				//float bsdf_pdf = bsdf->pdf(bsdfQuery);
				float light_pdf = light->pdf(emitterQuery);
				float wmat = last_bsdf_pdf / (light_pdf + last_bsdf_pdf);

				if (lastBsdfDelta)
					wmat = 1.0f;

				if (wmat > 0.0f)
				{
					//std::cout << wmat << endl;
					li += wmat*t*flight;
				}
				//specularInLastBounce = true;
			}

     		//successProb = std::min(0.9999f, t.maxCoeff());
     		//successProb = std::min(0.999f, t.maxCoeff());
     		successProb = 0.95;
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


			// get BSDF
			const BSDF* bsdf = its.mesh->getBSDF();
			Vector3f cam_dir = -traceRay.d;

			// sample bsdf
			BSDFQueryRecord bsdfquery(its.shFrame.toLocal(cam_dir)); // sets wi
			bsdfquery.measure = ESolidAngle;
			bsdfquery.uv = its.uv;

            bsdf->sampleDx(bsdfquery, its, *sampler);
			if (bsdfquery.probe)
            {
                while (!scene->rayIntersect(bsdfquery.probeRay, bsdfquery.its_out, false, its.mesh))
                    //break;
                    bsdf->sampleDx(bsdfquery, its, *sampler);

                bsdfquery.r = (bsdfquery.its_out.p - its.p).norm();
            }
            else
            {
                bsdfquery.its_out = its;
                bsdfquery.r = 0;
            }
            Color3f f = bsdf->sample(bsdfquery, its, sampler->next2D(), sampler->next2D());

			//Vector3f dx = bsdfquery.dx;

			last_bsdf_pdf = bsdf->pdf(bsdfquery);
			lastBsdfDelta = bsdfquery.isDelta;
			Vector3f next_light_dir = bsdfquery.its_out.toWorld(bsdfquery.wo).normalized();

			// we indirectly might hit a light source
			// EMS PART	
			{
				const Emitter* light = scene->getRandomEmitter(sampler->next1D());

				if (!bsdfquery.isDelta)
				{
					// prepare light source and create shadowRay 
					EmitterQueryRecord emitterQuery(bsdfquery.its_out.p);
					Color3f liLoc = light->sample(emitterQuery, sampler->next2D());

					Vector3f light_dir = emitterQuery.wi;

					// check if the light actually hits the intersection point
					if (!scene->rayIntersect(emitterQuery.shadowRay))
					{
						//BSDFQueryRecord bsdfquery(its.shFrame.toLocal(light_dir), its.shFrame.toLocal(cam_dir), ESolidAngle);
						//BSDFQueryRecord bsdfquery(its.shFrame.toLocal(cam_dir), its.shFrame.toLocal(light_dir), ESolidAngle);
                       
                        // set direction to light in local coordinates at its.p + dx (at its_out)
						bsdfquery.wo = bsdfquery.its_out.shFrame.toLocal(light_dir);

						float bsdf_pdf = bsdf->pdf(bsdfquery);
						float light_pdf = light->pdf(emitterQuery);
						float wem = light_pdf / (light_pdf + bsdf_pdf);

						// angle between light ray and surface normal
						float costhetaSurface = light_dir.normalized().dot(bsdfquery.its_out.shFrame.n.normalized());

						//cout << costhetaSurface << endl;
						if (wem > 0)
						{
							li += t*wem*liLoc*bsdf->eval(bsdfquery)*fabs(costhetaSurface)*lights.size();
						}
					}
				}
			}

			// MATS

            //BSDFQueryRecord bsdfQuery(its.shFrame.toLocal(wi));
			//bsdfQuery.uv = its.uv;

            //bsdf->sampleDx(bsdfQuery, its, sampler->next2D());


			// set new wi
			traceRay.d = next_light_dir;
			traceRay.o = bsdfquery.its_out.p;
			traceRay.mint = 1e-3;
			traceRay.maxt = std::numeric_limits<float>::infinity();
			traceRay.update();

			t *= f;
		}

		return li;
    }

    std::string toString() const {
        return "PathMisIntegrator[]";
    }
};

NORI_REGISTER_CLASS(PathMisIntegrator, "path_mis");
NORI_NAMESPACE_END
