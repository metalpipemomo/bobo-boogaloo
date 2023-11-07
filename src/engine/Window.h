#pragma once

#include "bpch.h"

#include <GLFW/glfw3.h>
//#include <glad/glad.h>

#include "Input.h"
//#include "Model/ModelLoader.h"
//#include "Audio/Audio.h"
#include "Time.h"
#include "EntityComponent/SceneManager.h"
#include "Coroutine/CoroutineScheduler.h"
#include "Physics/Physics.h"
//#include "Renderer/Camera.h"
//#include "Renderer/TextureLoader.h"
//#include "Renderer/ShaderLoader.h"
//#include "Renderer/Renderer.h"

struct WindowProperties
{
    unsigned int width, height;
    std::string name;

    WindowProperties(
        unsigned int width = 640,
        unsigned int height = 480,
        const std::string &name = "Default Window")
        : width(width), height(height), name(name) {}
};

class Window
{
public:
    Window(const WindowProperties &props = WindowProperties{})
    {
        m_Props = props;

        // Init stuff here, order matters
        Log::Init();
        Init();
        Input::Init(p_Window);
        Time::Init();
        Physics::Init();
        //Camera::Init((float)GetWidth() / (float)GetHeight());
        //TextureLoader::Init();
        //ShaderLoader::Init();
        //Renderer::Init();
        SceneManager::Init();
        //ModelLoader::Init();
        //Audio::Init();
        CoroutineScheduler::Init();
    }

    ~Window()
    {
        glfwDestroyWindow(p_Window);
    }

    void EventLoop()
    {
        while (!glfwWindowShouldClose(p_Window))
        {
            // System Frame Updates
            //Renderer::Update();
            Time::Update();
           // Audio::Update();
            CoroutineScheduler::Update();
            Physics::Update();
            // System Fixed Updates
            if (Time::DidFixedUpdate())
            {
            }

            // Draw
            //Renderer::Draw();

            // Window Updates
            Update();

            // Funny exit thing, also input test
            if (Input::GetKey(GLFW_KEY_L) &&
                Input::GetKey(GLFW_KEY_M) &&
                Input::GetKey(GLFW_KEY_A) &&
                Input::GetKey(GLFW_KEY_O))
            {
                exit(-1);
            }
        }
    }

    inline unsigned int GetWidth() const
    {
        return m_Props.width;
    }

    inline unsigned int GetHeight() const
    {
        return m_Props.height;
    }

private:
    void Init()
    {
       

    }

    void Update()
    {
        glfwPollEvents();
        glfwSwapBuffers(p_Window);
    }

    GLFWwindow *p_Window;
    WindowProperties m_Props;
};