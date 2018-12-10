#include <nori/normalModifier.h>
#include <nori/texture.h>

NORI_NAMESPACE_BEGIN

/**
 * \brief BumpMap / Lambertian BRDF model
 */
class BumpMap : public NormalModifier {
public:
    BumpMap(const PropertyList &propList) : m_bump_map(nullptr) {

    }
    virtual ~BumpMap() {
        delete m_bump_map;
    }

    /// Add texture for the bump_map
    virtual void addChild(NoriObject *obj) override {
        switch (obj->getClassType()) {
            case ETexture:
                if(obj->getIdName() == "bump_map") {
                    if (m_bump_map)
                        throw NoriException("There is already an bump_map defined!");
                    m_bump_map = static_cast<Texture<float> *>(obj);
                }
                else {
                    throw NoriException("The name of this texture does not match any field!");
                }
                break;

            default:
                throw NoriException("BumpMap::addChild(<%s>) is not supported!",
                                    classTypeName(obj->getClassType()));
        }
    }

    virtual void activate() override {
        if(!m_bump_map) {
            throw NoriException("There is no texture with name bump_map given!");
        }
    }

    virtual void adjustNormal(Vector3f& normal, const Point2f& uv) const override 
    {
        //float val = m_bump_map->eval(uv);
        //normal.x() = 0;
        //normal.normalize();
        normal = m_bump_map->eval(uv);
    }

    /// Return a human-readable summary
    virtual std::string toString() const override {
        return tfm::format(
            "BumpMap[]"
        );
    }

private:
    Texture<float> *m_bump_map;
};

NORI_REGISTER_CLASS(BumpMap, "bump_map");
NORI_NAMESPACE_END
