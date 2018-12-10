//
// Created by lukas on 02.07.17.
//

#include <glad/glad.h>
#include <iostream>
#include "ShaderProgramm.h"
#include <glm/glm.hpp>

static const char* vertex_shader_text =
"#version 330\n"
"in vec2 vPos;\n"
"out vec2 texcoord;\n"
"void main()\n"
"{\n"
"    gl_Position = vec4(vPos, 0.0, 1.0);\n"
"    texcoord = vPos;\n"
"}\n";

static const char* fragment_shader_text =
"#version 330\n"
"uniform sampler2D texture;\n"
"in vec2 texcoord;\n"
"out vec4 color;\n"
"float toSRGB(float value) {\n"
"    if (value < 0.0031308)\n"
"        return 12.92 * value;\n"
"    return 1.055 * pow(value, 0.41666) - 0.055;\n"
"}\n"
"void main()\n"
"{\n"
"    color = texture2D(texture, (texcoord+vec2(1.0,1.0))/2);\n"
"    color = vec4(toSRGB(color.r), toSRGB(color.g), toSRGB(color.b), 1);\n"
"}\n";


ShaderProgramm::ShaderProgramm() {
    const char* vShader_file = vertex_shader_text;
    const char* fShader_file = fragment_shader_text;


    unsigned int vShader = glCreateShader(GL_VERTEX_SHADER);
    unsigned int fShader = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(vShader, 1, &vShader_file, NULL);
    glShaderSource(fShader, 1, &fShader_file, NULL);

    glCompileShader(vShader);
    glCompileShader(fShader);


    int  success;
    char infoLog[512];
    glGetShaderiv(vShader, GL_COMPILE_STATUS, &success);
    unsigned int shad = vShader;
    if(success){
        shad = fShader;
        glGetShaderiv(fShader, GL_COMPILE_STATUS, &success);
    }
    if(!success)
    {
        glGetShaderInfoLog(shad, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
        exit(-1);
    }

    id = glCreateProgram();
    glAttachShader(id, vShader);
    glAttachShader(id, fShader);
    glLinkProgram(id);

    glDeleteShader(vShader);
    glDeleteShader(fShader);

}
void ShaderProgramm::use()
{
    glUseProgram(id);
}
// utility uniform functions
// ------------------------------------------------------------------------
void ShaderProgramm::set(const std::string &name, bool value) const
{
    glUniform1i(glGetUniformLocation(id, name.c_str()), (int)value);
}
// ------------------------------------------------------------------------
void ShaderProgramm::set(const std::string &name, int value) const
{
    glUniform1i(glGetUniformLocation(id, name.c_str()), value);
}
// ------------------------------------------------------------------------
void ShaderProgramm::set(const std::string &name, float value) const
{
    glUniform1f(glGetUniformLocation(id, name.c_str()), value);
}
// ------------------------------------------------------------------------
void ShaderProgramm::set(const std::string &name, const glm::vec2 &value) const
{
    glUniform2fv(glGetUniformLocation(id, name.c_str()), 1, &value[0]);
}
void ShaderProgramm::set(const std::string &name, float x, float y) const
{
    glUniform2f(glGetUniformLocation(id, name.c_str()), x, y);
}
// ------------------------------------------------------------------------
void ShaderProgramm::set(const std::string &name, const glm::vec3 &value) const
{
    glUniform3fv(glGetUniformLocation(id, name.c_str()), 1, &value[0]);
}
void ShaderProgramm::set(const std::string &name, float x, float y, float z) const
{
    glUniform3f(glGetUniformLocation(id, name.c_str()), x, y, z);
}
// ------------------------------------------------------------------------
void ShaderProgramm::set(const std::string &name, const glm::vec4 &value) const
{
    glUniform4fv(glGetUniformLocation(id, name.c_str()), 1, &value[0]);
}
void ShaderProgramm::set(const std::string &name, float x, float y, float z, float w)
{
    glUniform4f(glGetUniformLocation(id, name.c_str()), x, y, z, w);
}
// ------------------------------------------------------------------------
void ShaderProgramm::set(const std::string &name, const glm::mat2 &mat) const
{
    glUniformMatrix2fv(glGetUniformLocation(id, name.c_str()), 1, GL_FALSE, &mat[0][0]);
}
// ------------------------------------------------------------------------
void ShaderProgramm::set(const std::string &name, const glm::mat3 &mat) const
{
    glUniformMatrix3fv(glGetUniformLocation(id, name.c_str()), 1, GL_FALSE, &mat[0][0]);
}
// ------------------------------------------------------------------------
void ShaderProgramm::set(const std::string &name, const glm::mat4 &mat) const
{
    glUniformMatrix4fv(glGetUniformLocation(id, name.c_str()), 1, GL_FALSE, &mat[0][0]);
}
ShaderProgramm::~ShaderProgramm() {
    glDeleteProgram(id);

}



