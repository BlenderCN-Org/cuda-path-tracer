/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Wenzel Jakob, Romain Prévost

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

#if !defined(__NORI_SCENE_H)
#define __NORI_SCENE_H

#include <nori/bvh.h>
#include <nori/emitter.h>
#include "independent.h"
#include "enviromentmap.h"

NORI_NAMESPACE_BEGIN

/**
 * \brief Main scene data structure
 *
 * This class holds information on scene objects and is responsible for
 * coordinating rendering jobs. It also provides useful query routines that
 * are mostly used by the \ref Integrator implementations.
 */
class Scene : public NoriObject {
public:

    virtual size_t getSize() const  override  {return sizeof(Scene);}
    /// Construct a new scene object
    Scene(const PropertyList &);

    /// Release all memory
    virtual ~Scene();

    /// Return a pointer to the scene's kd-tree
    __device__ __host__  const BVH *getBVH() const { return &m_bvh; }

    /// Return a pointer to the scene's integrator
    __device__ __host__  const Integrator *getIntegrator() const { return m_integrator; }

    /// Return a pointer to the scene's integrator
    __device__ __host__  Integrator *getIntegrator() { return m_integrator; }

    /// Return a pointer to the scene's camera
    __device__ __host__  const Camera *getCamera() const { return m_camera; }

    /// Return a pointer to the scene's sample generator (const version)
    __device__ __host__  const Independent *getSampler() const { return m_sampler; }

    /// Return a pointer to the scene's sample generator
    __device__ __host__  Independent *getSampler() { return m_sampler; }

    /// Return a reference to an array containing all shapes
    __device__ __host__  const CombinedVector<Shape *> &getShapes() const { return m_shapes; }

    /// Return a reference to an array containing all lights
    __device__ __host__ const CombinedVector<Emitter*> &getLights() const { return m_emitters; }

    /// Return a random emitter
    __device__ __host__  const Emitter * getRandomEmitter(float rnd) const {
        auto const & n = m_emitters.size();
        size_t index = stdmin(
                static_cast<size_t>(std::floor(n*rnd)),
                n-1);
        return m_emitters[index];
    }

    __device__ const EnviromentMap* getEnvMap() const {return m_envmap;};

    /**
     * \brief Intersect a ray against all triangles stored in the scene
     * and return detailed intersection information
     *
     * \param ray
     *    A 3-dimensional ray data structure with minimum/maximum
     *    extent information
     *
     * \param its
     *    A detailed intersection record, which will be filled by the
     *    intersection query
     *
     * \return \c true if an intersection was found
     */
    __device__ bool rayIntersect(const Ray3f &ray, Intersection &its) const {
        return m_bvh.rayIntersect(ray, its, false);
    }

    /**
     * \brief Intersect a ray against all triangles stored in the scene
     * and \a only determine whether or not there is an intersection.
     *
     * This method much faster than the other ray tracing function,
     * but the performance comes at the cost of not providing any
     * additional information about the detected intersection
     * (not even its position).
     *
     * \param ray
     *    A 3-dimensional ray data structure with minimum/maximum
     *    extent information
     *
     * \return \c true if an intersection was found
     */
    __device__ bool rayIntersect(const Ray3f &ray) const {
        Intersection its; /* Unused */
        return m_bvh.rayIntersect(ray, its, true);
    }


    __device__ bool rayIntersect(const Ray3f &ray, Intersection &its, bool shadowRay, const Shape* mask) const {
        return m_bvh.rayIntersect(ray, its, shadowRay, mask);
    }

    /**
     * \brief Return an axis-aligned box that bounds the scene
     */
    const BoundingBox3f &getBoundingBox() const {
        return m_bvh.getBoundingBox();
    }

    /**
     * \brief Inherited from \ref NoriObject::activate()
     *
     * Initializes the internal data structures (kd-tree,
     * emitter sampling data structures, etc.)
     */
    virtual void activate() override;

    /// Add a child object to the scene (meshes, integrators etc.)
    virtual void addChild(NoriObject *obj) override;

    /// Return a string summary of the scene (for debugging purposes)
    virtual std::string toString() const override;
    virtual void gpuTransfer(NoriObject **objects) override;
    virtual EClassType getClassType() const override { return EScene; }
private:
    CombinedVector<Shape *> m_shapes;
    Integrator *m_integrator = nullptr;
    Independent *m_sampler = nullptr;
    EnviromentMap *m_envmap = nullptr;
    Camera *m_camera = nullptr;
    BVH m_bvh;
    CombinedVector<Emitter *> m_emitters;
 
public:
    float filter_h;
    float filter_sigma;
    float filter_r;
    float filter_f;


};

NORI_NAMESPACE_END

#endif /* __NORI_SCENE_H */
