//
// Created by lukas on 02.07.17.
//

#ifndef OPENGL_SAMPLE_MAINWINDOW_H
#define OPENGL_SAMPLE_MAINWINDOW_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "ShaderProgramm.h"
#include "CudaRender.h"
#include <GL/gl.h>
/**
 * MainWindow class, contains the opengl draw loop, checks for key inputs and fetches the console (the last thing mainly because the console
 * needs to be able to do OpenGL stuff and this forces it to be in the same thread therefor be part of the while loop)
 */
class MainWindow {
public:
    /**
     * Sets up the OpenGL windows (initalizing glfw, opening the windows, etc ...)
     */
    MainWindow();

    ~MainWindow();
    /**
     * Main loop, draws everything in an infinite loop
     * @param pConsole Console object to fetch commands from
     */
    void mainLoop();
    void saveView(std::string filename);
    void processInput();
    void loadFile(char *i);


private:
    ShaderProgramm* shader;
    GLFWwindow* window;
    CudaRender* cudaRender ; 
    GLuint textureID;
    GLuint vertex_buffer;

};


#endif //OPENGL_SAMPLE_MAINWINDOW_H
