#include <nori/integrator.h>
#include <nori/scene.h>
#include <nori/warp.h>
#include <nori/independent.h>
#include <nori/diffuse.h>
#include <nori/arealight.h>
#include <nori/dielectric.h>
//#include <nori/bsdf.h>
//#include <iostream>
#include <nori/pathMisIntegrator.h>


NORI_NAMESPACE_BEGIN


    PathMisIntegrator::PathMisIntegrator(const PropertyList &props) {
        integratorType = EPathMis;
    }

    __device__ Color3f PathMisIntegrator::Li(const Scene *scene, Sampler& sampler, Ray3f &ray,Color3f& t,bool& lastBsdfDelta,float& last_bsdf_pdf) const {

        /* Find the surface that is visible in the requested direction */

		size_t lights = scene->getLights().size();
        if(scene->getEnvMap())
            lights++;
		// initialize return color
		Color3f li(0.0f);
		float successProb = 0.99;
		//const BSDF* last_bsdf = NULL;

        Intersection its;
        Intersection its_out;

		//bool lastBsdfDelta = true;
		//float last_bsdf_pdf = 1.0f;
		//bool firstRun = true;

		for (int i=0;i<1;i++)
		{

            // ENV MAP
			// trace ray
            //If we did not intersect anything, we will intersect the Enviroment map if existent
            //This is the mats parts of it [as we can not reflect from the map again this function will still return]
			if (!scene->rayIntersect(ray, its)){
				//if(firstRun)
				//		asm("exit;");
                if(scene->getEnvMap()){
                   const  EnviromentMap* e = scene->getEnvMap();

                    EmitterQueryRecord emitterQuery(ray.o);
                    emitterQuery.wi = ray.d;
                    Color3f flight = CallMap(e,eval,emitterQuery);

                    //BSDFQueryRecord bsdfQuery(shFrame.toLocal(light_dir), its.shFrame.toLocal(cam_dir), ESolidAngle);
                    //float bsdf_pdf = bsdf->pdf(bsdfQuery);
                    float light_pdf = CallMap(e,pdf,emitterQuery);
                    //printf("%f\n",light_pdf);

                    //return li;
                    float wmat = last_bsdf_pdf / (light_pdf + last_bsdf_pdf);
                    //printf("%f\n",light_pdf);
                    if (lastBsdfDelta)
                        wmat = 1.0f;

                    if (wmat > 0.0f)
                    {
                        //std::cout << wmat << endl;
                        li += wmat*t*flight;
                    }
                }
				t(0) = -1;
                break;
			}
            


            // we directly hit a light source
            //printf("Color:%f\n",li.x());
			//firstRun = false;
			if (its.mesh->getEmitter() != NULL)
			{
                const Emitter* light = its.mesh->getEmitter();

				EmitterQueryRecord emitterQuery(ray.o, its.p, its.shFrame.n);
				Color3f flight = CallEmitter(light,eval,emitterQuery);

				//BSDFQueryRecord bsdfQuery(shFrame.toLocal(light_dir), its.shFrame.toLocal(cam_dir), ESolidAngle);
				//float bsdf_pdf = bsdf->pdf(bsdfQuery);
				float light_pdf = CallEmitter(light,pdf,emitterQuery);
                //return li;
				float wmat = last_bsdf_pdf / (light_pdf + last_bsdf_pdf);
                //printf("%f\n",light_pdf);
				if (lastBsdfDelta)
					wmat = 1.0f;

				if (wmat > 0.0f)
				{
					//std::cout << wmat << endl;
					li += wmat*t*flight;
				}
                //printf("%f %f %f\n",light_pdf,wmat,flight.x());
                //t(0) = -1;
                //break;
			}

     		//successProb = stdmin(0.99f, t.maxCoeff());
     		successProb = 0.95f;
			float f = CallSampler(sampler,next1D);
            //printf("%f %p \n",f,sampler);
            if (f > successProb)
			{
                t(0) = -1;
				break;
			}
			else
			{
				t /= successProb;
			}


			// MATS

			const BSDF* bsdf = its.mesh->getBSDF();
			Vector3f wi = -ray.d;

			BSDFQueryRecord bsdfQuery(its.shFrame.toLocal(wi));
			bsdfQuery.measure = ESolidAngle;
			bsdfQuery.uv = its.uv;

            CallBsdf(bsdf, sampleDx, bsdfQuery, its, sampler);
			if (bsdfQuery.probe)
            {
                // bssrdf case
                while (!scene->rayIntersect(bsdfQuery.probeRay, its_out, false, its.mesh))
                    //break;
                    CallBsdf(bsdf, sampleDx, bsdfQuery, its, sampler);

                bsdfQuery.probeDistance = (its_out.p - its.p).norm();
            }
            else
           {
                // brdf case
                its_out = its;
            }
			Color3f f_mat = CallBsdf(bsdf,sample,bsdfQuery, CallSampler(sampler,next2D));
			last_bsdf_pdf = CallBsdf(bsdf,pdf,bsdfQuery);

			lastBsdfDelta = bsdfQuery.isDelta;

			//Vector3f light_dir = its_out.toWorld(bsdfQuery.wo);
			//Vector3f cam_dir = -traceRay.d; // last cam direction
			Vector3f light_dir = its_out.toWorld(bsdfQuery.wo).normalized();
			Vector3f cam_dir = -ray.d; // last cam direction

			// set new wi
			//light_dir.y() += float(0.2);
			ray.d = light_dir;
            ray.o = its_out.p;
            ray.mint = float(1e-3);
            ray.update();
			//printf("ds:%f\n",last_bsdf_pdf);

			// we indirectly might hit a light source
			// EMS PART	
			{
				if (!bsdfQuery.isDelta)
				{
                    Color3f liLoc;
                    float light_pdf;
                    // prepare light source and create shadowRay
                    EmitterQueryRecord emitterQuery(its_out.p);

                    //sample env map
                    if(scene->getEnvMap()&&CallSampler(sampler,next1D)>float(lights-1)/lights){
                        const  EnviromentMap* e = scene->getEnvMap();
                        light_pdf = CallMap(e,pdf,emitterQuery);
                        liLoc = CallMap(e, sample, emitterQuery, CallSampler(sampler, next2D));
                    }else {
                        const Emitter *light = scene->getRandomEmitter(CallSampler(sampler, next1D));
                        liLoc = CallEmitter(light, sample, emitterQuery, CallSampler(sampler, next2D));
                        light_pdf = CallEmitter(light,pdf,emitterQuery);
                    }
                    Vector3f light_dir = emitterQuery.wi;
					// check if the light actually hits the intersection point
					if (!scene->rayIntersect(emitterQuery.shadowRay))
					{
						//BSDFQueryRecord bsdfQuery(its.shFrame.toLocal(light_dir), its.shFrame.toLocal(cam_dir), ESolidAngle);
						bsdfQuery.wo = its_out.shFrame.toLocal(light_dir);

						// bsdf based color
						float bsdf_pdf = CallBsdf(bsdf,pdf,bsdfQuery);

						//printf("%f %f\n",liLoc.x(),f.x());

						float wem = light_pdf / (light_pdf + bsdf_pdf);

						// angle between light ray and surface normal
						float costhetaSurface = light_dir.dot(its_out.shFrame.n);

						//cout << costhetaSurface << endl;
						if (wem > 0)
						{
                            Color3f bsdf_val = CallBsdf(bsdf,eval,bsdfQuery);
							li += t*wem*liLoc*bsdf_val*fabs(costhetaSurface)*lights;
						}
					}
				}
			}


			t *= f_mat;

		}
        //t(0) = -1;
		return li;
    }

    std::string PathMisIntegrator::toString() const {
        return "PathMisIntegrator[]";
    }

#ifndef __CUDA_ARCH__
    NORI_REGISTER_CLASS(PathMisIntegrator, "path_mis");
#endif
NORI_NAMESPACE_END
