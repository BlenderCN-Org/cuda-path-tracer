#include <nori/integrator.h>
#include <nori/scene.h>
#include <nori/warp.h>
#include <nori/bsdf.h>
#include <nori/vector.h>
#include <iostream>

NORI_NAMESPACE_BEGIN

class PathMatsIntegrator : public Integrator {
public:

	float maxLen;

    PathMatsIntegrator(const PropertyList &props) {

    }

    Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray) const {

        /* Find the surface that is visible in the requested direction */

		Ray3f traceRay = ray;
		std::vector<Emitter*> lights = scene->getLights();

		// initialize return color
		Color3f li(0.0f);
		Color3f t = Color3f(1.0f);
		float successProb = 0.94;
		
		while (true)
		{
			// trace ray
			Intersection its;
			if (!scene->rayIntersect(traceRay, its)) break;

			// we directly hit a light source
			if (its.mesh->getEmitter() != NULL)
			{
				const Emitter* light = its.mesh->getEmitter();

				EmitterQueryRecord emitterQuery(traceRay.o, its.p, its.shFrame.n);
				li += t*light->eval(emitterQuery);
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

			const BSDF* bsdf = its.mesh->getBSDF();
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

		return li;
    }

    std::string toString() const {
        return "PathMatsIntegrator[]";
    }
};

NORI_REGISTER_CLASS(PathMatsIntegrator, "path_mats");
NORI_NAMESPACE_END
