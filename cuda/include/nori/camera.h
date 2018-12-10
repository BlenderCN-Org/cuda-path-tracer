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

#if !defined(__NORI_CAMERA_H)
#define __NORI_CAMERA_H

#include <nori/object.h>

NORI_NAMESPACE_BEGIN

/**
 * \brief Generic camera interface
 * 
 * This class provides an abstract interface to cameras in Nori and
 * exposes the ability to sample their response function. By default, only
 * a perspective camera implementation exists, but you may choose to
 * implement other types (e.g. an environment camera, or a physically-based 
 * camera model that simulates the behavior actual lenses)
 */
class Camera : public NoriObject {
public:
    virtual size_t getSize() const  override  {return sizeof(Camera);}
    enum cameraType {EPerspective} cameraType;
    /**
     * \brief Importance sample a ray according to the camera's response function
     *
     * \param ray
     *    A ray data structure to be filled with a position 
     *    and direction value
     *
     * \param samplePosition
     *    Denotes the desired sample position on the film
     *    expressed in fractional pixel coordinates
     *
     * \param apertureSample
     *    A uniformly distributed 2D vector that is used to sample
     *    a position on the aperture of the sensor if necessary.
     *
     * \return
     *    An importance weight associated with the sampled ray.
     *    This accounts for the difference in the camera response
     *    function and the sampling density.
     */
    /*Color3f sampleRay(Ray3f &ray,
        const Point2f &samplePosition,
        const Point2f &apertureSample) const = 0;
    */
    /// Return the size of the output image in pixels
     const Vector2i &getOutputSize() const { return m_outputSize; }

    /// Return the camera's reconstruction filter in image space
    //const ReconstructionFilter *getReconstructionFilter() const { return m_rfilter; }

    /**
     * \brief Return the type of object (i.e. Mesh/Camera/etc.) 
     * provided by this instance
     * */
    virtual EClassType getClassType() const override { return ECamera; }
protected:
    Vector2i m_outputSize;
    ReconstructionFilter *m_rfilter;
};

/*
 * Need to add all Camera here for usage because no (working) virtual function
 */

#define CallCamera(object,function,...) CallCuda1(nori::Camera,cameraType,object,function,const,EPerspective,PerspectiveCamera,__VA_ARGS__)
NORI_NAMESPACE_END

#endif /* __NORI_CAMERA_H */
