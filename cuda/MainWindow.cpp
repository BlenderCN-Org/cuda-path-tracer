//
// Created by lukas on 02.07.17.
//

#include <iostream>
#include <glm/ext.hpp>
#include <highgui.h>
#include "MainWindow.h"
#include <GL/gl.h>
#include <nori/vector.h>

#define WX  1280
#define WY  720
//#define WX  800
//#define WY  600

static const float vertices[2*4] =
{
     -1.f, -1.f ,
     1.f, -1.f ,
     1.f, 1.f ,
     -1.f, 1.f
};




MainWindow::MainWindow() {
    if(!glfwInit()){
        exit(-1);
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    window = glfwCreateWindow(WX,WY,"OpenGL Nori", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        exit(-1);
    }




    glfwMakeContextCurrent(window);
    gladLoadGLLoader((GLADloadproc) glfwGetProcAddress);






    //initialize shader

    shader = new  ShaderProgramm();


    //add triangles for rendering
    glGenBuffers(1, &vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    //use shader, no need to set texture, as OpenGL uses the default one anyway
    shader->use();
    shader->set("texture",0);
    //insert VertexArray + VertexAttribArray
    GLuint vao=0;
    glGenVertexArrays(1,&vao);
    glBindVertexArray(vao);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE,
                          2*sizeof(vertices[0]), (void*) 0);
    //initialize Cuda
    this->cudaRender = new  CudaRender();

}



void MainWindow::mainLoop() {

    while (!glfwWindowShouldClose(this->window))
    {
            cudaRender->step();
        // input
        // -----
        glfwPollEvents();
        processInput();
        glfwPollEvents();
        // render
        // ------
        glClearColor(1.f, 1.f, 1.f, 0.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        shader->use();
        //glViewport(0,0,WX,WY);
	    glBindVertexArray(vertex_buffer);
        glDrawArrays(GL_TRIANGLE_FAN, 0,4);
        glBindVertexArray(0);	
        glfwSwapBuffers(window);

   }

}

//future add some controls
void MainWindow::processInput()
{


}

//future save view to file
void MainWindow::saveView(std::string filename) {

    int size = WX*WY;

    //allocate space for the buffer
    unsigned char* viewBuffer = (unsigned char *) malloc(3 * size);
    if(viewBuffer) {
        glReadPixels(0,0,WX,WY,GL_BGR,GL_UNSIGNED_BYTE,viewBuffer);
        cv::Mat image(cv::Size(size,size), CV_8UC3, viewBuffer, cv::Mat::AUTO_STEP);
        cv::flip(image,image,0);
        cv::imwrite(filename,image);
        free(viewBuffer);
    }


}
MainWindow::~MainWindow(){
    //not needed at
    delete cudaRender;
    delete shader;
    delete window;

}

void MainWindow::loadFile(char *t) {
    auto v = cudaRender->loadFile(t);
    //init static image
    //Generate Texture

    glGenTextures(1,&textureID);
    glBindTexture(GL_TEXTURE_2D,textureID);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,v.x(),v.y(),0,GL_RGBA,GL_FLOAT,NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    //glfwSetWindowSize(window,v.x(),v.y());
    //
    float wH = float(v.y())/v.x();
    if(WX*wH>WY)
        glViewport(0,0,WY/wH,WY);
    else
        glViewport(0,0,WX,WX*wH);

    cudaRender->registerTexture(textureID);
}
