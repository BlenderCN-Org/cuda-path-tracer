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

#if !defined(__NORI_COLOR_H)
#define __NORI_COLOR_H

#include <nori/common.h>

NORI_NAMESPACE_BEGIN

/**
 * \brief Represents a linear RGB color value
 */
struct Color3f : public Eigen::Array3f {
public:
    typedef Eigen::Array3f Base;

    /// Initialize the color vector with a uniform value
    __host__ __device__ Color3f(float value = 0.f) : Base(value, value, value) { }

    /// Initialize the color vector with specific per-channel values
    __host__ __device__  Color3f(float r, float g, float b) : Base(r, g, b) { }

    /// Construct a color vector from ArrayBase (needed to play nice with Eigen)
    template <typename Derived> __host__ __device__  Color3f(const Eigen::ArrayBase<Derived>& p)
        : Base(p) { }

    /// Assign a color vector from ArrayBase (needed to play nice with Eigen)
    template <typename Derived> __host__ __device__  Color3f &operator=(const Eigen::ArrayBase<Derived>& p) {
        this->Base::operator=(p);
        return *this;
    }

    /// Return a reference to the red channel
    __host__ __device__  float &r() { return x(); }
    /// Return a reference to the red channel (const version)
    __host__ __device__  const float &r() const { return x(); }
    /// Return a reference to the green channel
    __host__ __device__ float &g() { return y(); }
    /// Return a reference to the green channel (const version)
    __host__ __device__ const float &g() const { return y(); }
    /// Return a reference to the blue channel
    __host__ __device__ float &b() { return z(); }
    /// Return a reference to the blue channel (const version)
    __host__ __device__ const float &b() const { return z(); }

    /// Clamp to the positive range
    __host__ __device__ Color3f clamp() const { return Color3f(fmax(r(), 0.0f),
        fmax(g(), 0.0f), fmax(b(), 0.0f)); }

    /// Check if the color vector contains a NaN/Inf/negative value
    __host__ __device__ bool isValid() const;

    /// Convert from sRGB to linear RGB
    __host__ __device__  Color3f toLinearRGB() const;

    /// Convert from linear RGB to sRGB
    __host__ __device__  Color3f toSRGB() const;

    /// Return the associated luminance
    __device__  float getLuminance() const;

    /// Return a human-readable string summary
    __host__ std::string toString() const {
        return tfm::format("[%f, %f, %f]", coeff(0), coeff(1), coeff(2));
    }
};

/**
 * \brief Represents a linear RGB color and a weight
 *
 * This is used by Nori's image reconstruction filter code
 */
struct Color4f : public Eigen::Array4f {
public:
    typedef Eigen::Array4f Base;

    /// Create an zero value
    Color4f() : Base(0.0f, 0.0f, 0.0f, 0.0f) { }

    /// Create from a 3-channel color
    Color4f(const Color3f &c) : Base(c.r(), c.g(), c.b(), 1.0f) { }

    /// Initialize the color vector with specific per-channel values
    Color4f(float r, float g, float b, float w) : Base(r, g, b, w) { }

    /// Construct a color vector from ArrayBase (needed to play nice with Eigen)
    template <typename Derived> Color4f(const Eigen::ArrayBase<Derived>& p) 
        : Base(p) { }

    /// Assign a color vector from ArrayBase (needed to play nice with Eigen)
    template <typename Derived> Color4f &operator=(const Eigen::ArrayBase<Derived>& p) {
        this->Base::operator=(p);
        return *this;
    }

    /// Divide by the filter weight and convert into a \ref Color3f value
    Color3f divideByFilterWeight() const {
        if (w() != 0)
            return head<3>() / w();
        else
            return Color3f(0.0f);
    }

    /// Return a human-readable string summary
    std::string toString() const {
        return tfm::format("[%f, %f, %f, %f]", coeff(0), coeff(1), coeff(2), coeff(3));
    }
};

NORI_NAMESPACE_END

#endif /* __NORI_COLOR_H */
