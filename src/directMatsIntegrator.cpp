#include <nori/integrator.h>
#include <nori/scene.h>
#include <nori/warp.h>
#include <nori/bsdf.h>
#include <nori/vector.h>
#include <iostream>

NORI_NAMESPACE_BEGIN

class DirectMatsIntegrator : public Integrator {
public:

	float maxLen;

    DirectMatsIntegrator(const PropertyList &props) {

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

		BSDFQueryRecord bsdfQuery(shFrame.toLocal(cam_dir));
		bsdfQuery.uv = uv;
		Color3f f = bsdf->sample(bsdfQuery, its, sampler->next2D(), sampler->next2D());

		Vector3f light_dir_local = bsdfQuery.wo;
		Vector3f light_dir = its.toWorld(bsdfQuery.wo);

		Ray3f refl(its.p, light_dir);
		refl.mint = 1e-6;

		if (scene->rayIntersect(refl, its))
		{
			if (its.mesh->getEmitter() != NULL)
			{
				const Emitter* light = its.mesh->getEmitter();

				EmitterQueryRecord emitterQuery(refl.o, its.p, its.shFrame.n);

				li += f*light->eval(emitterQuery);
			}
		}

		return li;
    }

    std::string toString() const {
        return "DirectMatsIntegrator[]";
    }
};

NORI_REGISTER_CLASS(DirectMatsIntegrator, "direct_mats");
NORI_NAMESPACE_END
