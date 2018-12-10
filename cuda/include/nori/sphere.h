//
// Created by lukas on 04.12.17.
//

#ifndef NORIGL_SPHERE_H
#define NORIGL_SPHERE_H
NORI_NAMESPACE_BEGIN
    class Sphere : public Shape {
    public:

        __host__ size_t getSize()  const override {return sizeof(Sphere);}
        __host__ uint32_t getPrimitiveCount() const { return 1; }
         __host__ Sphere(const PropertyList & propList);
         __host__ BoundingBox3f getBoundingBox(uint32_t index) const ;
         __host__ __device__ Point3f getCentroid(uint32_t index) const  ;
         __device__ bool rayIntersect(uint32_t index, const Ray3f &ray, float &u, float &v, float &t) const  ;

        __device__ void setHitInformation(uint32_t index, const Ray3f &ray, Intersection & its) const ;

        __device__ void sampleSurface(ShapeQueryRecord & sRec, const Point2f & sample) const  ;
        __device__ float pdfSurface(const ShapeQueryRecord & sRec) const  ;

        __host__ virtual std::string toString() const override ;

    protected:
        Point3f m_position;
        float m_radius;
    };
NORI_NAMESPACE_END

#endif //NORIGL_SPHERE_H
