//
// Created by lukas on 26.11.17.
//

#ifndef NORIGL_PERSPECTIVE_H_H
#define NORIGL_PERSPECTIVE_H_H
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

#include <nori/camera.h>
//#include <nori/rfilter.h>
#include <nori/warp.h>
#include <Eigen/Geometry>

NORI_NAMESPACE_BEGIN

/**
 * \brief Perspective camera with depth of field
 *
 * This class implements a simple perspective camera model. It uses an
 * infinitesimally small aperture, creating an infinite depth of field.
 */
class PerspectiveCamera : public Camera {
    public:
    __host__    virtual size_t getSize() const override {return sizeof(PerspectiveCamera);} ;
    __host__    PerspectiveCamera(const PropertyList &propList);

    __host__    virtual void activate() override ;

    __device__ Color3f sampleRay(Ray3f &ray,
                          const Point2f &samplePosition,
                          const Point2f &apertureSample) const ;

    __host__  virtual void addChild(NoriObject *obj) override ;

        /// Return a human-readable summary
    __host__  virtual std::string toString() const override ;
    private:
        Vector2f m_invOutputSize;
        Transform m_sampleToCamera;
        Transform m_cameraToWorld;
        float m_fov;
        float m_nearClip;
        float m_farClip;
};

NORI_NAMESPACE_END
#endif //NORIGL_PERSPECTIVE_H_H
