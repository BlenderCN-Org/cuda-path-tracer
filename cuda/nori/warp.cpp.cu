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

#include <nori/warp.h>
#include <nori/independent.h>
//class Independent;
NORI_NAMESPACE_BEGIN

// CYLINDER of height 1
__device__ __host__ Vector3f Warp::squareToUniformCylinder(const Point2f& sample)
{
	float phi = 2*M_PI*sample.x();
	float z = sample.y();
	return Vector3f(cos(phi), sin(phi), z);
}


__device__ Vector3f Warp::sampleUniformHemisphere(Sampler sampler, const Normal3f &pole) {
    // Naive implementation using rejection sampling
    Vector3f v;
    do {
        v.x() = 1.f - 2.f * CallSampler(sampler,next1D);
        v.y() = 1.f - 2.f * CallSampler(sampler,next1D);
        v.z() = 1.f - 2.f * CallSampler(sampler,next1D);
    } while (v.squaredNorm() > 1.f);

    if (v.dot(pole) < 0.f)
        v = -v;
    v /= v.norm();

    return v;
}

// SQUARE
__device__ __host__  Point2f Warp::squareToUniformSquare(const Point2f &sample) {
    return sample;
}

__device__ __host__  float Warp::squareToUniformSquarePdf(const Point2f &sample) {
    return (sample.minCoeff() >= 0 && sample.maxCoeff() <= 1) ? 1.0f : 0.0f;
}

// DISK
__device__ __host__ Point2f Warp::squareToUniformDisk(const Point2f &sample) {
    float r = sqrtf(sample.x());
	float theta = 2*M_PI*sample.y();
	return Point2f(r*cos(theta), r*sin(theta));
}

__device__ __host__ float Warp::squareToUniformDiskPdf(const Point2f &p) {
	float r = p.norm();
	if (r<=1.0f)
	{
		return 1.0f/M_PI;
	}
	else return 0;
}

// CAP
__device__ __host__ Vector3f Warp::squareToUniformSphereCap(const Point2f &sample, float cosThetaMax) {
	Vector3f cyl = Warp::squareToUniformCylinder(sample);
	float offset = cosThetaMax;
	float height = 1.0f - offset;

	float z = offset + cyl.z()*height;
	float r = sqrtf(1-(z*z));
	return Vector3f(cyl.x()*r, cyl.y()*r, z);
}

__device__ __host__ float Warp::squareToUniformSphereCapPdf(const Vector3f &v, float cosThetaMax) {

	//float z = offset + cyl.z()*height;

	if (v.z() < cosThetaMax)
	{
		return 0;
	}
	else
	{
		float h = 1.0f - cosThetaMax;
		float area = 2.0f*M_PI*h;
		return 1.0f/area; // evenly distributed over area of the cap
	}
}

// SPHERE
__device__ __host__ Vector3f Warp::squareToUniformSphere(const Point2f &sample) {
	Vector3f cyl = Warp::squareToUniformCylinder(sample);
	float z = 2.0f*cyl.z()-1.0f;
	float r = sqrtf(1.0f-(z*z));
	return Vector3f(cyl.x()*r, cyl.y()*r, z);
}

__device__ __host__ float Warp::squareToUniformSpherePdf(const Vector3f &v) {
	return 1.0f/(4.0f*M_PI); // evenly distributed over area of a sphere
}


// HEMISPHERE
__device__ __host__ Vector3f Warp::squareToUniformHemisphere(const Point2f &sample) {
	return Warp::squareToUniformSphereCap(sample, 0);
}

__device__ __host__ float Warp::squareToUniformHemispherePdf(const Vector3f &v) {
	return squareToUniformSphereCapPdf(v, 0);
}


// COSINE HEMISPHERE
__device__ __host__ Vector3f Warp::squareToCosineHemisphere(const Point2f &sample) {
	Point2f d = Warp::squareToUniformDisk(sample);
	return Vector3f(d.x(), d.y(), sqrtf(1.0f - d.x()*d.x() - d.y()*d.y()));
}

__device__ __host__ float Warp::squareToCosineHemispherePdf(const Vector3f &v) {

	if (v.z() >= 0)
		return v.z()/M_PI;
	else 
		return 0;
	
}


// BECKMANN
__device__ __host__ Vector3f Warp::squareToBeckmann(const Point2f &sample, float alpha) {

    float theta = acos(sqrtf(1.0f / ( 1.0f - alpha*alpha*log( 1.0f - sample.x()))));
	float phi = sample.y()*2.0f*M_PI;

	return Vector3f(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta));

}

__device__ __host__ float Warp::squareToBeckmannPdf(const Vector3f &m, float alpha) {

	//return 0;

	if (m.z() >= 0 && fabs(m.norm()-1) < float(1e-6))
	{
		float theta = acos(m.z());
		float phi = atan2(m.y(), m.x());

		float alpha2 = alpha*alpha;
		return exp(-powf(tan(theta), 2) / alpha2) / (M_PI*alpha2*powf(cos(theta), 3));
	}
	else return 0;

}


__device__ Vector3f Warp::squareToUniformTriangle(const Point2f &sample) {
	float su1 = sqrtf(sample.x());
    float u = 1.f - su1, v = sample.y() * su1;
    return Vector3f(u,v,1.f-u-v);
}

NORI_NAMESPACE_END
