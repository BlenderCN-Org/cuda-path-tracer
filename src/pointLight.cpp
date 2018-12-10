//#include <nori/shape.h>
#include <nori/emitter.h>
//#include <nori/warp.h>

NORI_NAMESPACE_BEGIN

class PointLight : public Emitter {

protected:
	Color3f m_power;
	Point3f m_position;

public:
    PointLight(const PropertyList & propList) {
        m_position = propList.getPoint3("position", Point3f());
        m_power = propList.getColor("power", Color3f());
    }

    /**
     * \brief Sample the emitter and return the importance weight (i.e. the
     * value of the Emitter divided by the probability density
     * of the sample with respect to solid angles).
     *
     * \param lRec    An emitter query record (only ref is needed)
     * \param sample  A uniformly distributed sample on \f$[0,1]^2\f$
     *
     * \return The emitter value divided by the probability density of the sample.
     *         A zero value means that sampling failed.
     */
    Color3f sample(EmitterQueryRecord &lRec, const Point2f &sample) const
	{
		Vector3f dir = (m_position - lRec.ref);

		lRec.wi = dir.normalized();
		lRec.n = -lRec.wi;

		lRec.shadowRay.o = m_position;
		lRec.shadowRay.d = -lRec.wi;
		lRec.shadowRay.mint = 0;
		lRec.shadowRay.maxt = dir.norm() - 1e-3;
		
		return m_power / (4 * M_PI) / pdf(lRec);
	}

    /**
     * \brief Evaluate the emitter
     *
     * \param lRec
     *     A record with detailed information on the emitter query
     * \return
     *     The emitter value, evaluated for each color channel
     */
    Color3f eval(const EmitterQueryRecord &lRec) const
	{
		//return m_power/lRec.n.dot(-lRec.wi)*(lRec.ref-lRec.p).squaredNorm();
		return m_power/pdf(lRec);
	}

    /**
     * \brief Compute the probability of sampling \c lRec.p.
     *
     * This method provides access to the probability density that
     * is realized by the \ref sample() method.
     *
     * \param lRec
     *     A record with detailed information on the emitter query
     *
     * \return
     *     A probability/density value
     */
    float pdf(const EmitterQueryRecord &lRec) const
	{
		return (lRec.ref-lRec.p).squaredNorm();
	}


    /// Sample a photon
    Color3f samplePhoton(Ray3f &ray, const Point2f &sample1, const Point2f &sample2) const {
        throw NoriException("Emitter::samplePhoton(): not implemented!");
    }


    /**
     * \brief Virtual destructor
     * */
    ~PointLight() {}


    virtual std::string toString() const
	{
		return "PointLight[power="+ m_power.toString() + ", pos=" + m_position.toString() + "]";
	}

};

NORI_REGISTER_CLASS(PointLight, "point");
NORI_NAMESPACE_END
