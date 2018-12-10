//
// Created by lukas on 02.07.17.
//

#ifndef OPENGL_SHADERPROGRAMM_H
#define OPENGL_SHADERPROGRAMM_H

#include <string>
#include <sstream>
#include <cstring>
#include <fstream>
#include <glm/vec3.hpp>
#include <glm/detail/type_mat2x2.hpp>
class ShaderProgramm {
public:
    ShaderProgramm();
    ~ShaderProgramm();
    void use();

    void set(const std::string &name, bool value) const;


    void set(const std::string &name, int value) const;

    void set(const std::string &name, float value) const;

    void set(const std::string &name, float x, float y) const;

    void set(const std::string &name, float x, float y, float z) const;

    void set(const std::string &name, float x, float y, float z, float w);

    void set(const std::string &name, const glm::vec3 &value) const;

    void set(const std::string &name, const glm::vec2 &value) const;

    void set(const std::string &name, const glm::vec4 &value) const;

    void set(const std::string &name, const glm::mat2 &mat) const;

    void set(const std::string &name, const glm::mat3 &mat) const;

    void set(const std::string &name, const glm::mat4 &mat) const;
private:
    unsigned int id;


};


#endif //OPENGL_SHADERPROGRAMM_H
