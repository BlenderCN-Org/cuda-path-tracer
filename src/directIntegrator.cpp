#include <nori/integrator.h>
#include <nori/scene.h>
#include <nori/warp.h>
#include <nori/bsdf.h>

NORI_NAMESPACE_BEGIN

class DirectIntegrator : public Integrator {
public:

	float maxLen;

    DirectIntegrator(const PropertyList &props) {

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
				continue;

			BSDFQueryRecord bsdfQuery(shFrame.toLocal(light_dir), shFrame.toLocal(cam_dir), ESolidAngle);
			bsdfQuery.uv = uv;

			// bsdf based color
			Color3f f = bsdf->eval(bsdfQuery);

			// angle between light ray and normal
			float costheta = light_dir.normalized().dot(its.shFrame.n.normalized());

			// integration formula
			li += liLoc*f*fabs(costheta);
		}

		return li;
    }

    std::string toString() const {
        return "DirectIntegrator[]";
    }
};

NORI_REGISTER_CLASS(DirectIntegrator, "direct");
NORI_NAMESPACE_END
