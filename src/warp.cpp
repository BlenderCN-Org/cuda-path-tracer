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
#include <nori/vector.h>
#include <nori/frame.h>

NORI_NAMESPACE_BEGIN

// CYLINDER of height 1
Vector3f Warp::squareToUniformCylinder(const Point2f& sample)
{
	float phi = 2*M_PI*sample.x();
	float z = sample.y();
	return Vector3f(cos(phi), sin(phi), z);
}

Vector3f Warp::sampleUniformHemisphere(Sampler *sampler, const Normal3f &pole) {
    // Naive implementation using rejection sampling
    Vector3f v;
    do {
        v.x() = 1.f - 2.f * sampler->next1D();
        v.y() = 1.f - 2.f * sampler->next1D();
        v.z() = 1.f - 2.f * sampler->next1D();
    } while (v.squaredNorm() > 1.f);

    if (v.dot(pole) < 0.f)
        v = -v;
    v /= v.norm();

    return v;
}

// SQUARE
Point2f Warp::squareToUniformSquare(const Point2f &sample) {
    return sample;
}

float Warp::squareToUniformSquarePdf(const Point2f &sample) {
    return ((sample.array() >= 0).all() && (sample.array() <= 1).all()) ? 1.0f : 0.0f;
}

// DISK
Point2f Warp::squareToUniformDisk(const Point2f &sample) {
    float r = std::sqrt(sample.x());
	float theta = 2*M_PI*sample.y();
	return Point2f(r*cos(theta), r*sin(theta));
}

float Warp::squareToUniformDiskPdf(const Point2f &p) {
	float r = p.norm();
	if (r<=1.0f)
	{
		return 1.0f/M_PI;
	}
	else return 0;
}


// DISK exp decay normed with second exp decay
// http://library.imageworks.com/pdfs/imageworks-library-BSSRDF-sampling.pdf
Point2f Warp::squareToExpDecayDisk(float v, const Point2f &sample) {

    float v2 = 2.0f*v;
    float Rm = sqrt(v*12.46);

    //float r = -log(std::sqrt(sample.x()))/sigma;
	float theta = 2.0f*M_PI*sample.y();
    float r = sqrt(-v2*log(1.0-sample.x()*(1.0f-exp(-Rm*Rm/v2))));
	return Point2f(r*cos(theta), r*sin(theta));
}

float Warp::squareToExpDecayDiskPdf(float v, const Point2f &p) {
	float r2 = p.squaredNorm();
    float Rm = sqrt(v*12.46);
    float Rm2 = Rm*Rm;

    float v2 = 2.0f*v;

	if (r2<=Rm*Rm)
	{
		//return (sigma*sigma*exp(-sigma*r))/M_PI;
		float Rd = 1.0f/(v2*M_PI)*exp(-r2/v2) /M_PI;
        return Rd/(1.0f-exp(-Rm*Rm/v2));
	}
	else return 0;
}


const float A = 0.8;
//const float s = 3.5f + 100.0f*pow((A - 0.33f), 4);
const float s = 1;

float diffuseScatterApproxCDF(float dmfp, float r)
{
    float rsd = -r*s/dmfp;
    return 1.0f - 0.25f*exp(rsd)-0.75f*exp(rsd/3.0f);
}

float diffuseScatterApproxCDFDerivative(float dmfp, float r)
{
    float sd = s/dmfp;
    float rsd = -r*sd;
    return sd * 0.25f* (exp(rsd) + exp(rsd/3.0f));
}

Point2f Warp::squareToSubsurfaceRd(float dmfp, const Point2f& sample2)
{
    float sample = sample2.x();

    // newton find inverse of CDF
    float err = 1.0f;
    float sdInv = dmfp/s;
    float r0 = -log((1.0f - sample) * 4.0/3.0)*3.0f*sdInv;
    float r1 = -log((1.0f - sample) * 4.0)*3.0f*sdInv;
    
    float err0 = fabs(diffuseScatterApproxCDF(dmfp, r0) - sample);;
    float err1 = fabs(diffuseScatterApproxCDF(dmfp, r1) - sample);

    if (err0 < err1)  r1=r0;
    else r1 = r0;

    //(cout << "|phi - cdf(r)|_0a=" << fabs(diffuseScatterApproxCDF(r0) - sample) << endl;
    //cout << "|phi - cdf(r)|_0b=" << fabs(diffuseScatterApproxCDF(r1) - sample) << endl;

    for (int i=0; i<5; ++i)
    {
        float f = (diffuseScatterApproxCDF(dmfp, r0) - sample);
        float df = diffuseScatterApproxCDFDerivative(dmfp, r0);
        r1 = r0 - f/df;
        r0 = r1;
    }

    //cout << "|phi - cdf(r)|_1=" << fabs(diffuseScatterApproxCDF(dmfp, r0) - sample) << endl;
	
	// r0 = CDF^-1(psi)

    float theta = 2.0f*M_PI*sample2.y();

    return Point2f(r0*cos(theta), r0*sin(theta));
}

float Warp::squareToSubsurfaceRdPDF(float dmfp, const Point2f& p)
{
    //float rmax = sqrt(dmfp/s*12.46);
    float r = p.norm();
    return s*(exp(-s*r/dmfp) + exp(-s*r/(3.0f*dmfp)))/(8.0f*M_PI*dmfp*r);
    //return s*(exp(-s*r/dmfp) + exp(-s*r/(3.0f*dmfp)))/(8.0f*M_PI*dmfp*r)*2.0f*M_PI*r;
    //return -0.25*exp(-r*s/dmfp) - 0.75*exp(-r*s/(3.0f*dmfp));
    //return diffuseScatterApproxCDFDerivative(dmfp, r) / (2.0f*M_PI*r);
}

// CAP
Vector3f Warp::squareToUniformSphereCap(const Point2f &sample, float cosThetaMax) {
	Vector3f cyl = Warp::squareToUniformCylinder(sample);
	float offset = cosThetaMax;
	float height = 1.0f - offset;

	float z = offset + cyl.z()*height;
	float r = std::sqrt(1-(z*z));
	return Vector3f(cyl.x()*r, cyl.y()*r, z);
}

float Warp::squareToUniformSphereCapPdf(const Vector3f &v, float cosThetaMax) {

	float offset = cosThetaMax;
	float height = 1.0f - offset;
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
Vector3f Warp::squareToUniformSphere(const Point2f &sample) {
	Vector3f cyl = Warp::squareToUniformCylinder(sample);
	float z = 2.0f*cyl.z()-1.0f;
	float r = std::sqrt(1.0f-(z*z));
	return Vector3f(cyl.x()*r, cyl.y()*r, z);
}

float Warp::squareToUniformSpherePdf(const Vector3f &v) {
	return 1.0f/(4.0f*M_PI); // evenly distributed over area of a sphere
}


// HEMISPHERE
Vector3f Warp::squareToUniformHemisphere(const Point2f &sample) {
	return Warp::squareToUniformSphereCap(sample, 0);
}

float Warp::squareToUniformHemispherePdf(const Vector3f &v) {
	return squareToUniformSphereCapPdf(v, 0);
}


// COSINE HEMISPHERE
Vector3f Warp::squareToCosineHemisphere(const Point2f &sample) {
	Point2f d = Warp::squareToUniformDisk(sample);
	return Vector3f(d.x(), d.y(), std::sqrt(1.0f - d.x()*d.x() - d.y()*d.y()));
}

float Warp::squareToCosineHemispherePdf(const Vector3f &v) {

	if (v.z() >= 0)
		return v.z()/M_PI;
	else 
		return 0;
	
}


// BECKMANN
Vector3f Warp::squareToBeckmann(const Point2f &sample, float alpha) {

    float theta = acos(std::sqrt(1.0f / ( 1.0f - alpha*alpha*log( 1.0f - sample.x()))));
	float phi = sample.y()*2.0f*M_PI;

	return Vector3f(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta));

}

float Warp::squareToBeckmannPdf(const Vector3f &m, float alpha) {

	//return 0;

	if (m.z() >= 0 && fabs(m.norm()-1) < 1e-6)
	{
		float theta = acos(m.z());
		float phi = atan2(m.y(), m.x());

		float alpha2 = alpha*alpha;
		return exp(-std::pow(tan(theta), 2) / alpha2) / (M_PI*alpha2*std::pow(cos(theta), 3));
	}
	else return 0;

}


Vector3f Warp::squareToUniformTriangle(const Point2f &sample) {
    float su1 = sqrtf(sample.x());
    float u = 1.f - su1, v = sample.y() * su1;
    return Vector3f(u,v,1.f-u-v);
}

NORI_NAMESPACE_END
