#include <nori/integrator.h>
#include <nori/scene.h>
#include <nori/warp.h>

NORI_NAMESPACE_BEGIN

class AverageVisibilityIntegrator : public Integrator {
public:

	float maxLen;

    AverageVisibilityIntegrator(const PropertyList &props) {

        maxLen = props.getFloat("length");
    }

    Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray) const {
        /* Find the surface that is visible in the requested direction */
        Intersection its;
        if (!scene->rayIntersect(ray, its))
            return Color3f(1.0f);

		// create ray in random direction on hemisphere
		Vector3f d = Warp::sampleUniformHemisphere(sampler, its.shFrame.n);
		Ray3f refl(its.p, d, 1e-6, this->maxLen);

		if (scene->rayIntersect(refl, its))
		{
			// occlusion case
			return Color3f(0.0f);
		}
		else
		{
			// no hit on segment
			return Color3f(1.0f);
		}
    }

    std::string toString() const {
        return "AverageVisibilityIntegrator[]";
    }
};

NORI_REGISTER_CLASS(AverageVisibilityIntegrator, "av");
NORI_NAMESPACE_END
