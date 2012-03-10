/*
 * Copyright (c) 2012 Matthew Arsenault
 *
 * This file is part of Milkway@Home.
 *
 * Milkyway@Home is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Milkyway@Home is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Milkyway@Home.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wfloat-equal"
#pragma GCC diagnostic ignored "-Wtype-limits"
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include <glm/glm.hpp>
#include <glm/ext.hpp>

#pragma GCC diagnostic pop

#include "nbody_graphics.h"
#include "nbody_gl.h"
#include "nbody_config.h"
#include "milkyway_util.h"

#include <freetype-gl.h>
#include <vertex-buffer.h>

#include <assert.h>

#include <errno.h>
#include <cstring>
#include <cstdlib>

#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include <streambuf>
#include <ios>
#include <stdexcept>
#include <stack>

#if !BOINC_APPLICATION
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <fcntl.h>
  #include <errno.h>
#endif /* !BOINC_APPLICATION */

#define GLFW_INCLUDE_GL3 1
//#define GLFW_NO_GLU 1
#include <GL/glfw3.h>

//#include <glload/gl_3_3.h>
#include <glutil/glutil.h>

#include "nbody_particle_texture.h"


extern "C" unsigned char particle_texture_vertex_glsl[];
extern "C" size_t particle_texture_vertex_glsl_len;

extern "C" unsigned char particle_texture_fragment_glsl[];
extern "C" size_t particle_texture_fragment_glsl_len;


extern "C" unsigned char axes_vertex_glsl[];
extern "C" size_t axes_vertex_glsl_len;

extern "C" unsigned char axes_fragment_glsl[];
extern "C" size_t axes_fragment_glsl_len;


extern "C" unsigned char galaxy_vertex_glsl[];
extern "C" size_t galaxy_vertex_glsl_len;

extern "C" unsigned char galaxy_fragment_glsl[];
extern "C" size_t galaxy_fragment_glsl_len;


extern "C" unsigned char text_vertex_glsl[];
extern "C" size_t text_vertex_glsl_len;

extern "C" unsigned char text_fragment_glsl[];
extern "C" size_t text_fragment_glsl_len;


static const float zNear = 0.01f;
static const float zFar = 1000.0f;

static glm::mat4 cameraToClipMatrix(1.0f);

static glm::mat4 textCameraToClipMatrix(1.0f);

static glutil::ViewData initialViewData =
{
	glm::vec3(0.0f, 0.0f, 0.0f),  // center position
    glm::fquat(1.0f, 0.0f, 0.0f, 0.0f), // view direction
	30.0f,  // radius
	0.0f    // spin rotation of up axis
};

static glutil::ViewScale viewScale =
{
	0.05f, 100.0f, // min, max view radius
	0.5f, 0.1f,   // radius deltas
    4.0f, 1.0f,
	90.0f / 250.0f // rotation scale
};

static glutil::ViewPole viewPole = glutil::ViewPole(initialViewData, viewScale, glutil::MB_LEFT_BTN);



static void nbglGetProgramLog(GLuint program, const char* name)
{
    GLint logLength = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
    if (logLength > 0)
    {
        GLint newlen;
        GLchar* log = new GLchar[logLength + 1];
        glGetProgramInfoLog(program, logLength, &newlen, log);
        fprintf(stderr,
                "Linker output (%s):\n"
                "--------------------------------------------------------------------------------\n"
                "%s\n"
                "--------------------------------------------------------------------------------\n",
                name,
                log);

        delete[] log;
    }

	GLint status = GL_FALSE;
	glGetProgramiv(program, GL_LINK_STATUS, &status);
	if (!status)
	{
        throw std::runtime_error("Linking shader program failed");
	}
}

static const char* showShaderType(GLenum type)
{
    switch (type)
    {
        case GL_VERTEX_SHADER:
            return "Vertex shader";
        case GL_FRAGMENT_SHADER:
            return "Fragment shader";
        default:
            return "<bad shader type>";
    }
}

static GLuint createShaderFromSrc(const char* src, GLint len, GLenum type)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, &len);
    glCompileShader(shader);

    GLint logLength = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
    if (logLength > 0)
    {
        GLchar* logBuf = new GLchar[logLength + 1];
        glGetShaderInfoLog(shader, logLength, NULL, logBuf);

        fprintf(stderr,
                "Shader compile log '%s':\n"
                "--------------------------------------------------------------------------------\n"
                "%s\n"
                "--------------------------------------------------------------------------------\n",
                showShaderType(type),
                logBuf);

        delete[] logBuf;
    }

    GLint status = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (!status)
    {
        throw std::runtime_error("Error compiling shader");
    }

    return shader;
}

static GLuint nbglCreateProgram(const char* name,
                                const char* vertSrc,
                                const char* fragSrc,
                                GLint vertSrcLen,
                                GLint fragSrcLen)
{
    GLuint vertexShader = createShaderFromSrc(vertSrc, (GLint) vertSrcLen, GL_VERTEX_SHADER);
    GLuint pixelShader = createShaderFromSrc(fragSrc, (GLint) fragSrcLen, GL_FRAGMENT_SHADER);

    GLuint program = glCreateProgram();

    glAttachShader(program, vertexShader);
    glAttachShader(program, pixelShader);

    glLinkProgram(program);

    glDetachShader(program, vertexShader);
    glDetachShader(program, pixelShader);

    glDeleteShader(vertexShader);
    glDeleteShader(pixelShader);

    nbglGetProgramLog(program, name);

    return program;
}



class GalaxyModel;

struct Color
{
    GLfloat r, g, b;
};

struct NBodyVertex
{
    GLfloat x, y, z;

    NBodyVertex() : x(0.0f), y(0.0f), z(0.0f) { }
    NBodyVertex(GLfloat x, GLfloat y, GLfloat z) : x(x), y(y), z(z) { }
};


class NBodyText
{
private:
    texture_font_t* font;
    texture_atlas_t* atlas;

    /* Text which will remain constant */
    vertex_buffer_t* constTextBuffer;

    /* Buffer of text which may change every frame */
    vertex_buffer_t* textBuffer;

    vec2 penEndConst;

    GLuint constTextVAO;
    GLuint textVAO;

    struct TextProgramData
    {
        GLuint program;

        GLint positionLoc;
        GLint texCoordLoc;
        GLint texPositionLoc;

        GLint textTextureLoc;

        GLint modelToCameraMatrixLoc;
        GLint cameraToClipMatrixLoc;
    } textProgram;

public:
    void prepareConstantText(const scene_t* scene);
    void prepareTextVAOs();
    void drawProgressText(const scene_t* scene);
    void loadFont();
    void loadShader();

    NBodyText();
    ~NBodyText();
};

class NBodyAxes
{
private:
    GLuint axesVAO;
    GLuint axesBuffer;
    GLuint axesColorBuffer;

    struct AxesProgramData
    {
        GLuint program;
        GLint positionLoc;
        GLint colorLoc;
        GLint modelToCameraMatrixLoc;
        GLint cameraToClipMatrixLoc;
    } axesProgramData;

public:
    void draw(const glm::mat4& modelMatrix);
    void loadShader();
    void createBuffers();
    void prepareVAO();

    NBodyAxes();
    ~NBodyAxes();
};

NBodyAxes::NBodyAxes()
{
    this->axesVAO = 0;
    this->axesBuffer = 0;
    this->axesColorBuffer = 0;

    this->axesProgramData.program = 0;
    this->axesProgramData.positionLoc = -1;
    this->axesProgramData.colorLoc = -1;
    this->axesProgramData.modelToCameraMatrixLoc = -1;
    this->axesProgramData.cameraToClipMatrixLoc = -1;
}

NBodyAxes::~NBodyAxes()
{
    if (!this->axesProgramData.program)
        glDeleteProgram(this->axesProgramData.program);

    glDeleteBuffers(1, &this->axesBuffer);
    glDeleteBuffers(1, &this->axesColorBuffer);
    glDeleteVertexArrays(1, &this->axesVAO);
}

void NBodyAxes::draw(const glm::mat4& modelMatrix)
{
    glUseProgram(this->axesProgramData.program);
    glUniformMatrix4fv(this->axesProgramData.modelToCameraMatrixLoc, 1, GL_FALSE, glm::value_ptr(modelMatrix));
    glUniformMatrix4fv(this->axesProgramData.cameraToClipMatrixLoc, 1, GL_FALSE, glm::value_ptr(cameraToClipMatrix));

    glBindVertexArray(this->axesVAO);
    glDrawArrays(GL_LINES, 0, 6);
    glBindVertexArray(0);
    glUseProgram(0);
}

void NBodyAxes::loadShader()
{
    this->axesProgramData.program = nbglCreateProgram("axes program",
                                                      (const char*) axes_vertex_glsl,
                                                      (const char*) axes_fragment_glsl,
                                                      (GLint) axes_vertex_glsl_len,
                                                      (GLint) axes_fragment_glsl_len);

    GLuint program = this->axesProgramData.program;

    this->axesProgramData.positionLoc = glGetAttribLocation(program, "position");
    this->axesProgramData.colorLoc = glGetAttribLocation(program, "inputColor");
    this->axesProgramData.modelToCameraMatrixLoc = glGetUniformLocation(program, "modelToCameraMatrix");
    this->axesProgramData.cameraToClipMatrixLoc = glGetUniformLocation(program, "cameraToClipMatrix");
}

#define AXES_LENGTH 10.0f

void NBodyAxes::createBuffers()
{
    static const GLfloat axes[6][4] =
        {
            { 0.0f,        0.0f,        0.0f,        1.0f },
            { AXES_LENGTH, 0.0f,        0.0f,        1.0f },
            { 0.0f,        0.0f,        0.0f,        1.0f },
            { 0.0f,        AXES_LENGTH, 0.0f,        1.0f },
            { 0.0f,        0.0f,        0.0f,        1.0f },
            { 0.0f,        0.0f,        AXES_LENGTH, 1.0f }
        };

    static const GLfloat colors[6][4] =
        {
            { 1.0f, 0.0f, 0.0f, 0.75f },
            { 1.0f, 0.0f, 0.0f, 0.75f },
            { 0.0f, 1.0f, 0.0f, 0.75f },
            { 0.0f, 1.0f, 0.0f, 0.75f },
            { 0.0f, 0.0f, 1.0f, 0.75f },
            { 0.0f, 0.0f, 1.0f, 0.75f }
        };

    glGenBuffers(1, &this->axesBuffer);
    glGenBuffers(1, &this->axesColorBuffer);

    glBindBuffer(GL_ARRAY_BUFFER, this->axesBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(axes), (const GLfloat*) axes, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, this->axesColorBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(colors), (const GLfloat*) colors, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void NBodyAxes::prepareVAO()
{
    glGenVertexArrays(1, &this->axesVAO);
    glBindVertexArray(this->axesVAO);
    glEnableVertexAttribArray(this->axesProgramData.positionLoc);
    glEnableVertexAttribArray(this->axesProgramData.colorLoc);

    glBindBuffer(GL_ARRAY_BUFFER, this->axesBuffer);
    glVertexAttribPointer(this->axesProgramData.positionLoc, 4, GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, this->axesColorBuffer);
    glVertexAttribPointer(this->axesProgramData.colorLoc, 4, GL_FLOAT, GL_FALSE, 0, 0);

    glBindVertexArray(0);
}

class NBodyGraphics
{
private:
    const scene_t* scene;
    GLFWwindow window;

    GLuint particleVAO;
    GLuint axesVAO;

    enum DrawMode
    {
        POINTS,
        MONOCHROME_POINTS,
        TEXTURED_SPRITES
    } drawMode;

    struct ParticleTextureProgramData
    {
        GLuint program;

        GLint positionLoc;
        GLint colorLoc;

        GLint modelToCameraMatrixLoc;
        GLint cameraToClipMatrixLoc;

        GLint particleTextureLoc;
    } particleTextureProgram;

    struct ParticlePointProgramData
    {
        GLuint program;

        GLint positionLoc;
        GLint colorLoc;

        GLint modelToCameraMatrixLoc;
        GLint cameraToClipMatrixLoc;

        GLint particleTextureLoc;
    } particlePointProgram;



    GLuint positionBuffer;
    GLuint velocityBuffer;
    GLuint accelerationBuffer;
    GLuint colorBuffer;
    GLuint particleTexture;

    bool running;

    GalaxyModel* galaxyModel;

    NBodyText text;
    NBodyAxes axes;

    void loadShaders();
    void createBuffers();
    void prepareVAOs();
    void createPositionBuffer();

public:

    struct DrawOptions
    {
        bool fullscreen;
        bool screensaverMode;
        bool paused;
        bool drawOrbitTrace;
        bool drawInfo;
        bool drawAxes;
        bool drawParticles;
        bool floatMode;
        bool cmCentered;
        bool drawHelp;
        bool monochromatic;

        float pointSize;
    } drawOptions;


    NBodyGraphics(const scene_t* scene);
    ~NBodyGraphics();

    void prepareContext();
    void populateBuffers();

    void loadModel(GalaxyModel& model);
    void loadColors();
    void drawAxes();
    void drawBodies(const glm::mat4& modelMatrix);
    void readSceneData();

    void display();
    void mainLoop();
    void stop()
    {
        this->running = false;
    }
};


// For access from callbacks
static NBodyGraphics* globalGraphicsContext = NULL;



static void errorHandler(int errorCode, const char* msg)
{
    fprintf(stderr, "GLFW error (%d): %s\n", errorCode, msg);
}

static void resizeHandler(GLFWwindow window, int w, int h)
{
    float wf = (float) w;
    float hf = (float) h;
    float aspectRatio = wf / hf;
    glutil::MatrixStack persMatrix;
    persMatrix.Perspective(90.0f, aspectRatio, zNear, zFar);
    cameraToClipMatrix = persMatrix.Top();

    const float fontHeight = 12.0f; // FIXME: hardcoded font height
    textCameraToClipMatrix = glm::ortho(0.0f, wf, -hf, 0.0f);
    textCameraToClipMatrix = glm::translate(textCameraToClipMatrix, glm::vec3(0.0f, -fontHeight, 0.0f));

    glViewport(0, 0, (GLsizei) w, (GLsizei) h);

    if (globalGraphicsContext)
    {
        globalGraphicsContext->display();
        glfwSwapBuffers();
    }
}

static int closeHandler(GLFWwindow window)
{
    if (globalGraphicsContext)
    {
        globalGraphicsContext->stop();
    }

    return 0;
}

static int getGLFWModifiers(GLFWwindow window)
{
    int modifiers = 0;

    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
    {
        modifiers |= glutil::MM_KEY_SHIFT;
    }

    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
    {
        modifiers |= glutil::MM_KEY_CTRL;
    }

    return modifiers;
}

static glutil::MouseButtons glfwButtonToGLUtil(int button)
{
    switch (button)
    {
        case GLFW_MOUSE_BUTTON_LEFT:
            return glutil::MB_LEFT_BTN;
        case GLFW_MOUSE_BUTTON_RIGHT:
            return glutil::MB_RIGHT_BTN;
        case GLFW_MOUSE_BUTTON_MIDDLE:
        default:
            return glutil::MB_MIDDLE_BTN;
    }
}

static void mouseButtonHandler(GLFWwindow window, int button, int action)
{
    int x, y;
    int modifiers = getGLFWModifiers(window);
    glfwGetMousePos(window, &x, &y);
    viewPole.MouseClick(glfwButtonToGLUtil(button), action == GLFW_PRESS, modifiers, glm::ivec2(x, y));
}

static void mousePosHandler(GLFWwindow window, int x, int y)
{
    viewPole.MouseMove(glm::ivec2(x, y));
}

static void scrollHandler(GLFWwindow window, int x, int y)
{
    int direction = y > 0;
    int modifiers = getGLFWModifiers(window);

    viewPole.MouseWheel(direction, modifiers, glm::ivec2(x, y));
}

static void keyHandler(GLFWwindow window, int key, int pressed)
{
    if (!pressed) // release
        return;

    NBodyGraphics* ctx = globalGraphicsContext;
    if (!ctx)
        return;

    NBodyGraphics::DrawOptions& opts = ctx->drawOptions;

    switch (key)
    {
        case GLFW_KEY_A:
            opts.drawAxes = !opts.drawAxes;
            break;

        case GLFW_KEY_I:
            opts.drawInfo = !opts.drawInfo;
            break;

        case GLFW_KEY_Q:
            ctx->stop();
            break;

        case GLFW_KEY_B:
            /*
            scene->starsize *= 1.1;
            if (scene->starsize > 100.0)
            {
                scene->starsize = 100.0;
            }
            */
            break;

        case GLFW_KEY_S:
            /*
            scene->starsize *= 0.9;
            if (scene->starsize < 1.0e-3)
            {
                scene->starsize = 1.0e-3;
            }
            */
            break;

        case GLFW_KEY_H:
      //case '?':
            //scene->drawHelp = !scene->drawHelp;
            break;

        case GLFW_KEY_O: /* Toggle camera following CM or on milkyway center */
            //scene->cmCentered = !scene->cmCentered;
            break;

        case GLFW_KEY_R: /* Toggle floating */
            //scene->floatMode = !scene->floatMode;
            break;

        case GLFW_KEY_P:
            ctx->drawOptions.paused = !ctx->drawOptions.paused;
            break;

        case GLFW_KEY_C:
            //scene->monochromatic = !scene->monochromatic;
            break;

        default:
            return;
    }
}

static void nbglSetHandlers(NBodyGraphics* graphicsContext)
{
    glfwSetErrorCallback(errorHandler);

    glfwSetWindowSizeCallback(resizeHandler);
    glfwSetWindowCloseCallback(closeHandler);

    glfwSetKeyCallback(keyHandler);
    //glfwSetKeyCallback(charHandler);
    glfwSetMouseButtonCallback(mouseButtonHandler);
    glfwSetMousePosCallback(mousePosHandler);
    glfwSetScrollCallback(scrollHandler);

    globalGraphicsContext = graphicsContext;
}


void NBodyText::loadShader()
{
    this->textProgram.program = nbglCreateProgram("text program",
                                                  (const char*) text_vertex_glsl,
                                                  (const char*) text_fragment_glsl,
                                                  (GLint) text_vertex_glsl_len,
                                                  (GLint) text_fragment_glsl_len);

    this->textProgram.positionLoc = glGetAttribLocation(this->textProgram.program, "position");
    this->textProgram.textTextureLoc = glGetUniformLocation(this->textProgram.program, "textTexture");
    this->textProgram.texPositionLoc = glGetAttribLocation(this->textProgram.program, "texPosition");
    this->textProgram.modelToCameraMatrixLoc = glGetUniformLocation(this->textProgram.program, "modelToCameraMatrix");
    this->textProgram.cameraToClipMatrixLoc = glGetUniformLocation(this->textProgram.program, "cameraToClipMatrix");
}

NBodyText::NBodyText()
{
    this->atlas = NULL;
    this->font = NULL;
    this->textBuffer = NULL;
    this->constTextBuffer = NULL;

    this->penEndConst.x = 0.0f;
    this->penEndConst.y = 0.0f;

    this->constTextVAO = 0;
    this->textVAO = 0;
}

NBodyText::~NBodyText()
{
    if (this->textProgram.program != 0)
        glDeleteProgram(this->textProgram.program);

    glDeleteVertexArrays(1, &this->constTextVAO);
    glDeleteVertexArrays(1, &this->textVAO);

    if (this->font)
    {
        texture_font_delete(this->font);
    }

    if (this->constTextBuffer)
    {
        vertex_buffer_delete(this->constTextBuffer);
    }

    if (this->textBuffer)
    {
        vertex_buffer_delete(this->textBuffer);
    }
}


typedef struct
{
    float x, y, z;    // position
    float s, t;       // texture
    float r, g, b, a; // color
} vertex_t;

static void add_text(vertex_buffer_t* buffer,
                     texture_font_t* font,
                     const wchar_t* text,
                     const vec4* color,
                     vec2* pen)
{
    float r = color->red, g = color->green, b = color->blue, a = color->alpha;
    float left = pen->x;
    const size_t n = wcslen(text);

    for (size_t i = 0; i < n; ++i)
    {
        if (text[i] == L'\n')
        {
            pen->x = left;
            pen->y -= (font->linegap + font->height);
            continue;
        }

        texture_glyph_t* glyph = texture_font_get_glyph(font, text[i]);
        if (glyph)
        {
            int kerning = 0;
            if (i > 0)
            {
                kerning = texture_glyph_get_kerning(glyph, text[i - 1]);
            }

            pen->x += kerning;
#if 0
            int x0  = (int) (pen->x + glyph->offset_x);
            int y0  = (int) (pen->y + glyph->offset_y);
            int x1  = (int) (x0 + glyph->width) ;
            int y1  = (int) (y0 - glyph->height);
#else
            float x0  = pen->x + glyph->offset_x;
            float y0  = pen->y + glyph->offset_y;
            float x1  = x0 + glyph->width;
            float y1  = y0 - glyph->height;
#endif

            float s0 = glyph->s0;
            float t0 = glyph->t0;
            float s1 = glyph->s1;
            float t1 = glyph->t1;
            GLuint index = buffer->vertices->size;

            GLuint indices[] =
                {
                    index, index + 1, index + 2,
                    index, index + 2, index + 3
                };

            vertex_t vertices[] =
                {
                    { x0, y0, 0.0f,  s0, t0,  r, g, b, a },
                    { x0, y1, 0.0f,  s0, t1,  r, g, b, a },
                    { x1, y1, 0.0f,  s1, t1,  r, g, b, a },
                    { x1, y0, 0.0f,  s1, t0,  r, g, b, a }
                };

            vertex_buffer_push_back_indices(buffer, indices, 6);
            vertex_buffer_push_back_vertices(buffer, vertices, 4);
            pen->x += glyph->advance_x;
        }
    }
}

static const vec4 x11green = { 0.0f, 1.0f, 0.0f, 0.5f };



    //glBlendFunc(GL_CONSTANT_COLOR_EXT, GL_ONE_MINUS_SRC_COLOR);
    //glBlendFunc(GL_DST_COLOR, GL_ONE_MINUS_SRC_ALPHA);
    //glBlendFunc(GL_DST_COLOR, GL_ONE);
    //glBlendFunc(GL_SRC_COLOR, GL_ONE);
//    glBlendFunc(GL_ONE, GL_ONE);
    //glBlendFunc(GL_SRC_COLOR, GL_ONE_MINUS_SRC_ALPHA);
    //glBindSampler(0, 0);

    //glEnable(GL_COLOR_MATERIAL);
    //glEnable(GL_BLEND);
    //float alpha = 0.5f;
    //glBlendColor(1.0f - alpha, 1.0f - alpha, 1.0f - alpha, 1.0f);



void NBodyText::drawProgressText(const scene_t* scene)
{
    wchar_t buf[1024];

    /* Start right after the constant portion */
    vec2 pen = this->penEndConst;

    swprintf(buf, sizeof(buf),
             L"Time: %4.3f / %4.3f Gyr (%4.3f %%)\n",
             scene->info.currentTime,
             scene->info.timeEvolve,
             100.0f * scene->info.currentTime / scene->info.timeEvolve
        );

    vertex_buffer_clear(this->textBuffer);
    add_text(this->textBuffer,
             this->font,
             buf,
             &x11green,
             &pen);

    glUseProgram(this->textProgram.program);
    glUniformMatrix4fv(this->textProgram.cameraToClipMatrixLoc, 1, GL_FALSE, glm::value_ptr(textCameraToClipMatrix));

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, this->atlas->id);
    glUniform1i(this->textProgram.textTextureLoc, 0);

    glBindVertexArray(this->constTextVAO);
    vertex_buffer_render(this->constTextBuffer, GL_TRIANGLES, "" /* "vtc" */);

    glBindVertexArray(this->textVAO);
    vertex_buffer_render(this->textBuffer, GL_TRIANGLES, "" /* "vtc" */);

    assert(glGetError() == GL_NO_ERROR);

    glUseProgram(0);
    glBindVertexArray(0);
}

NBodyGraphics::NBodyGraphics(const scene_t* scene)
{
    this->window = NULL;
    this->scene = scene;
    this->particleVAO = 0;
    this->axesVAO = 0;

    this->positionBuffer = 0;
    this->velocityBuffer = 0;
    this->accelerationBuffer = 0;
    this->colorBuffer = 0;

    this->particleTexture = 0;

    this->particleTextureProgram.program = 0;
    this->particleTextureProgram.positionLoc = -1;
    this->particleTextureProgram.colorLoc = -1;
    this->particleTextureProgram.modelToCameraMatrixLoc = -1;
    this->particleTextureProgram.cameraToClipMatrixLoc = -1;
    this->particleTextureProgram.particleTextureLoc = -1;

    this->running = false;

    this->drawOptions.fullscreen = false;
    this->drawOptions.screensaverMode = false;
    this->drawOptions.paused = false;
    this->drawOptions.drawOrbitTrace = true;
    this->drawOptions.drawInfo = true;
    this->drawOptions.drawAxes = true;
    this->drawOptions.drawParticles = true;
    this->drawOptions.floatMode = false;
    this->drawOptions.cmCentered = false;
    this->drawOptions.drawHelp = false;
    this->drawOptions.monochromatic = false;

    this->galaxyModel = NULL;
}

NBodyGraphics::~NBodyGraphics()
{
    GLuint buffers[4];

    if (this->particleTextureProgram.program != 0)
        glDeleteProgram(this->particleTextureProgram.program);

    buffers[0] = this->positionBuffer;
    buffers[1] = this->velocityBuffer;
    buffers[2] = this->accelerationBuffer;
    buffers[3] = this->colorBuffer;

    glDeleteBuffers(4, buffers);

    glDeleteVertexArrays(1, &this->particleVAO);
    glDeleteTextures(1, &this->particleTexture);
}

class GalaxyModel
{
private:
    NBodyVertex* points;
    Color* colors;
    GLuint nPoints;
    GLuint count;

    GLuint vao;
    GLuint buffer;
    GLuint colorBuffer;

    double bulgeScale;
    double diskScale;
    double totalHeight;

    double totalDiameter;
    GLint radialSlices;
    GLint axialSlices;
    double axialSliceSize;
    double diameterSlice;


    struct GalaxyProgramData
    {
        GLuint program;

        GLint positionLoc;
        GLint modelToCameraMatrixLoc;
        GLint cameraToClipMatrixLoc;
    } programData;

    NBodyText text;

    void makePoint(NBodyVertex& point, bool neg, double r, double theta);
    void generateSegment(bool neg);
    void generateJoiningSegment();

public:
    void generateModel();

    size_t size() const
    {
        return this->nPoints * sizeof(NBodyVertex);
    }

    size_t colorSize() const
    {
        return this->nPoints * sizeof(Color);
    }

    void draw(const glm::mat4& modelMatrix) const
    {
        glUseProgram(this->programData.program);
        glUniformMatrix4fv(this->programData.modelToCameraMatrixLoc, 1, GL_FALSE, glm::value_ptr(modelMatrix));
        glUniformMatrix4fv(this->programData.cameraToClipMatrixLoc, 1, GL_FALSE, glm::value_ptr(cameraToClipMatrix));

        glBindVertexArray(this->vao);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, this->nPoints);
        glBindVertexArray(0);
        glUseProgram(0);
    }

    void loadShaders()
    {
        this->programData.program = nbglCreateProgram("galaxy program",
                                                      (const char*) galaxy_vertex_glsl,
                                                      (const char*) galaxy_fragment_glsl,
                                                      (GLint) galaxy_vertex_glsl_len,
                                                      (GLint) galaxy_fragment_glsl_len);

        this->programData.positionLoc = glGetAttribLocation(this->programData.program, "position");
        this->programData.modelToCameraMatrixLoc = glGetUniformLocation(this->programData.program, "modelToCameraMatrix");
        this->programData.cameraToClipMatrixLoc = glGetUniformLocation(this->programData.program, "cameraToClipMatrix");
    }

    void prepareVAO()
    {
        glGenVertexArrays(1, &this->vao);

        glBindVertexArray(this->vao);
        glEnableVertexAttribArray(this->programData.positionLoc);
        glBindBuffer(GL_ARRAY_BUFFER, this->buffer);
        glVertexAttribPointer(this->programData.positionLoc, 3, GL_FLOAT, GL_FALSE, 0, 0);

        //glBindBuffer(GL_ARRAY_BUFFER, this->colorBuffer);
        //glVertexAttribPointer(this->programData.colorLoc, 3, GL_FLOAT, GL_FALSE, 0, 0);

        glBindVertexArray(0);
    }

    void bufferData()
    {
        assert(this->nPoints != 0);

        glGenBuffers(1, &this->buffer);
        glGenBuffers(1, &this->colorBuffer);

        glBindBuffer(GL_ARRAY_BUFFER, this->buffer);
        glBufferData(GL_ARRAY_BUFFER, this->size(), (const GLfloat*) this->points, GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glBindBuffer(GL_ARRAY_BUFFER, this->colorBuffer);
        glBufferData(GL_ARRAY_BUFFER, this->colorSize(), (const GLfloat*) this->colors, GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    GalaxyModel()
    {
        this->points = NULL;
        this->colors = NULL;
        this->nPoints = 0;
        this->count = 0;

        this->vao = 0;
        this->buffer = 0;
        this->colorBuffer = 0;


        this->bulgeScale = 0.7;
        this->diskScale = 4.0;
        this->totalHeight = 2.0 * this->bulgeScale;

        this->totalDiameter = 33.0;

        this->radialSlices = 50;
        GLuint nDiameterSlices = 2 * this->radialSlices + 1;
        this->diameterSlice = this->totalDiameter / (double) nDiameterSlices;

        this->axialSlices = 50;
        this->axialSliceSize = M_2PI / (double) this->axialSlices;

        this->programData.program = 0;
        this->programData.positionLoc = -1;
        this->programData.modelToCameraMatrixLoc = -1;
        this->programData.cameraToClipMatrixLoc = -1;
    }

    ~GalaxyModel()
    {
        if (this->programData.program != 0)
            glDeleteProgram(this->programData.program);

        glDeleteVertexArrays(1, &this->vao);

        glDeleteBuffers(1, &this->buffer);
        glDeleteBuffers(1, &this->colorBuffer);

        delete[] this->points;
        delete[] this->colors;
    }
};

void NBodyText::prepareConstantText(const scene_t* scene)
{
    vec2 pen;
    wchar_t buf[1024];

    pen.x = 0.0f;
    pen.y = 0.0f;

    swprintf(buf, sizeof(buf),
             L"N-body simulation (%d particles)\n",
             scene->nbody);

    add_text(this->constTextBuffer,
             this->font,
             buf,
             &x11green,
             &pen);

    this->penEndConst = pen;
}

void NBodyText::prepareTextVAOs()
{
    glGenVertexArrays(1, &this->constTextVAO);
    glBindVertexArray(this->constTextVAO);

    vertex_buffer_upload(this->constTextBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, this->constTextBuffer->vertices_id);
    glEnableVertexAttribArray(this->textProgram.positionLoc);
    glEnableVertexAttribArray(this->textProgram.texPositionLoc);
    glVertexAttribPointer(this->textProgram.positionLoc, 3, GL_FLOAT, GL_FALSE, sizeof(vertex_t), (void*) 0);
    glVertexAttribPointer(this->textProgram.texPositionLoc, 2, GL_FLOAT, GL_FALSE, sizeof(vertex_t), (void*) (3 * sizeof(GLfloat)));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->constTextBuffer->indices_id);

    glGenVertexArrays(1, &this->textVAO);
    glBindVertexArray(this->textVAO);

    vertex_buffer_upload(this->textBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, this->textBuffer->vertices_id);
    glEnableVertexAttribArray(this->textProgram.positionLoc);
    glEnableVertexAttribArray(this->textProgram.texPositionLoc);
    glVertexAttribPointer(this->textProgram.positionLoc, 3, GL_FLOAT, GL_FALSE, sizeof(vertex_t), (void*) 0);
    glVertexAttribPointer(this->textProgram.texPositionLoc, 2, GL_FLOAT, GL_FALSE, sizeof(vertex_t), (void*) (3 * sizeof(GLfloat)));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->textBuffer->indices_id);

    glBindVertexArray(0);
}

void NBodyText::loadFont()
{
    this->atlas = texture_atlas_new(512, 512, 3);
    if (!this->atlas)
    {
        throw std::runtime_error("Failed to load text atlas");
    }

    const char* fontPath = "/Users/matt/src/milkywayathome_client/freetype-gl/Vera.ttf";
    const float fontSize = 12.0f;
    this->font = texture_font_new(atlas, fontPath, fontSize);
    if (!this->font)
    {
        throw std::runtime_error("Failed to load font data");
    }

    this->textBuffer = vertex_buffer_new("v3f:t2f:c4f");
    this->constTextBuffer = vertex_buffer_new("v3f:t2f:c4f");
    if (!this->textBuffer || !this->constTextBuffer)
    {
        throw std::runtime_error("Failed to create text vertex buffers");
    }

    texture_font_load_glyphs(this->font,
                             L" !\"#$%&'()*+,-./0123456789:;<=>?"
                             L"@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_"
                             L"`abcdefghijklmnopqrstuvwxyz{|}~"
                             L" ");
    printf("Glyphs loaded %d\n", glGetError());
}

void NBodyGraphics::loadShaders()
{
    this->particleTextureProgram.program = nbglCreateProgram("particle texture program",
                                                             (const char*) particle_texture_vertex_glsl,
                                                             (const char*) particle_texture_fragment_glsl,
                                                             (GLint) particle_texture_vertex_glsl_len,
                                                             (GLint) particle_texture_fragment_glsl_len);

    GLuint program = this->particleTextureProgram.program;

    this->particleTextureProgram.positionLoc = glGetAttribLocation(program, "position");
    this->particleTextureProgram.colorLoc = glGetAttribLocation(program, "inputColor");
    this->particleTextureProgram.modelToCameraMatrixLoc = glGetUniformLocation(program, "modelToCameraMatrix");
    this->particleTextureProgram.cameraToClipMatrixLoc = glGetUniformLocation(program, "cameraToClipMatrix");
    this->particleTextureProgram.particleTextureLoc = glGetUniformLocation(program, "particleTexture");
}

void NBodyGraphics::prepareVAOs()
{
    glGenVertexArrays(1, &this->particleVAO);
    glBindVertexArray(this->particleVAO);
    glEnableVertexAttribArray(this->particleTextureProgram.positionLoc);
    glEnableVertexAttribArray(this->particleTextureProgram.colorLoc);

    glBindBuffer(GL_ARRAY_BUFFER, this->positionBuffer);
    /* 4th component is not included */
    glVertexAttribPointer(this->particleTextureProgram.positionLoc, 3, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), 0);

    glBindBuffer(GL_ARRAY_BUFFER, this->colorBuffer);
    glVertexAttribPointer(this->particleTextureProgram.colorLoc, 3, GL_FLOAT, GL_FALSE, 0, 0);

    glBindVertexArray(0);
}

void NBodyGraphics::readSceneData()
{
    const GLfloat* positions = (const GLfloat*) this->scene->rTrace;
    GLint nbody = this->scene->nbody;

    /*
    if (OPA_load_int(&this->scene->useSecondBuffer))
    {
        printf("Using second buffer\n");
        positions += nbody;
    }
    else
    {
        printf("Using first buffer\n");
    }
    */

    glBindBuffer(GL_ARRAY_BUFFER, this->positionBuffer);
    glBufferSubData(GL_ARRAY_BUFFER, 0, 4 * nbody * sizeof(GLfloat), positions);
}

void NBodyGraphics::drawBodies(const glm::mat4& modelMatrix)
{
    glUseProgram(this->particleTextureProgram.program);
    glUniformMatrix4fv(this->particleTextureProgram.modelToCameraMatrixLoc, 1, GL_FALSE, glm::value_ptr(modelMatrix));
    glUniformMatrix4fv(this->particleTextureProgram.cameraToClipMatrixLoc, 1, GL_FALSE, glm::value_ptr(cameraToClipMatrix));


    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, this->particleTexture);
    glUniform1i(this->particleTextureProgram.particleTextureLoc, 1);

    glBindVertexArray(this->particleVAO);

    glDepthMask(GL_FALSE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    //glBlendFunc(GL_ONE, GL_ONE);
    //float alpha = 0.5f;
    //glBlendFunc(GL_SRC_COLOR, GL_ONE_MINUS_SRC_ALPHA);
    //glBlendFunc(GL_DST_COLOR, GL_ONE_MINUS_SRC_ALPHA);

    //glBlendColor(1.0f - alpha, 1.0f - alpha, 1.0f - alpha, 1.0f);

    //glDepthMask(GL_FALSE);

    //printf("Pre arst %d\n", glGetError());
    //glEnable(GL_POINT_SPRITE);
    //printf("enable point sprite %d\n", glGetError());
    //glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
    //printf("tex envi %d\n", glGetError());

    glDrawArrays(GL_POINTS, 0, this->scene->nbody);


    //glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
    //glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);



    glBindVertexArray(0);
    glUseProgram(0);
}

void NBodyGraphics::createBuffers()
{
    GLuint buffers[4];

    glGenBuffers(4, buffers);

    this->positionBuffer = buffers[0];
    this->velocityBuffer = buffers[1];
    this->accelerationBuffer = buffers[2];
    this->colorBuffer = buffers[3];
}

void NBodyGraphics::createPositionBuffer()
{
    GLint nbody = this->scene->nbody;
    const GLfloat* positions = (const GLfloat*) this->scene->rTrace;

    glBindBuffer(GL_ARRAY_BUFFER, this->positionBuffer);
	glBufferData(GL_ARRAY_BUFFER, 4 * nbody * sizeof(GLfloat), positions, GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void NBodyGraphics::populateBuffers()
{
    this->createPositionBuffer();

    this->particleTexture = createParticleTexture(32);
}

void NBodyGraphics::loadModel(GalaxyModel& model)
{
    model.generateModel();
    model.bufferData();
    model.loadShaders();
    model.prepareVAO();
    this->galaxyModel = &model;
}

void NBodyGraphics::loadColors()
{
    GLint nbody = this->scene->nbody;

    /* assign random particle colors */
    srand((unsigned int) time(NULL));

    Color* color = new Color[nbody];

    for (GLint i = 0; i < nbody; ++i)
    {
        /*
        if (r[i].ignore)
        {
            R = grey.x;  // TODO: Random greyish color?
            G = grey.z;
            B = grey.y;
        }
        else
        {
            R = ((double) rand()) / ((double) RAND_MAX);
            G = ((double) rand()) / ((double) RAND_MAX) * (1.0 - R);
            B = 1.0 - R - G;
        }
        */

        double R = ((double) rand()) / ((double) RAND_MAX);
        double G = ((double) rand()) / ((double) RAND_MAX) * (1.0 - R);
        double B = 1.0 - R - G;

        double scale;
        if (R >= G && R >= B)
        {
            scale = 1.0 + ((double) rand()) / ((double) RAND_MAX) * (std::min(2.0, 1.0 / R) - 1.0);
        }
        else if (G >= R && G >= B)
        {
            scale = 1.0 + ((double) rand()) / ((double) RAND_MAX) * (std::min(2.0, 1.0 / G) - 1.0);
        }
        else
        {
            scale = 1.0 + ((double) rand()) / ((double) RAND_MAX) * (std::min(2.0, 1.0 / B) - 1.0);
        }

        //color[i].ignore = color[i].ignore;
        color[i].r = (GLfloat) R * scale;
        color[i].g = (GLfloat) G * scale;
        color[i].b = (GLfloat) B * scale;
    }

    glBindBuffer(GL_ARRAY_BUFFER, this->colorBuffer);
	glBufferData(GL_ARRAY_BUFFER, 3 * nbody * sizeof(GLfloat), color, GL_STATIC_DRAW);
    glVertexAttribPointer(this->particleTextureProgram.colorLoc, 3, GL_FLOAT, GL_FALSE, 0, 0);

    delete[] color;
}

void NBodyGraphics::prepareContext()
{
    this->createBuffers();
    this->loadShaders();
    this->prepareVAOs();

    this->text.loadShader();
    this->text.loadFont();
    this->text.prepareConstantText(this->scene);
    this->text.prepareTextVAOs();

    this->axes.loadShader();
    this->axes.createBuffers();
    this->axes.prepareVAO();

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClearDepth(1.0);


    glEnable(GL_MULTISAMPLE);


    // allow changing point size from within shader
    // as well as smoothing them to look more spherical
    //glEnable(GL_POINT_SMOOTH);
    //glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glEnable(GL_PROGRAM_POINT_SIZE);
    printf("pointsmooth %d\n", glGetError());

    //glPointSize(3.0f);
    glPointSize(10.0f);
    printf("Pointsize %d\n", glGetError());


    //glEnable(GL_CULL_FACE);
    //glCullFace(GL_BACK);
    //glFrontFace(GL_CW);

    float maxSmoothPointSize[2];
    glGetFloatv(GL_SMOOTH_POINT_SIZE_RANGE, (GLfloat*) &maxSmoothPointSize);
    printf("point size range %f %f\n", maxSmoothPointSize[0], maxSmoothPointSize[1]);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glDepthRange(0.0f, 1.0f);


    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


    //glEnable(GL_ALPHA_TEST);


    //glBlendFunc(GL_CONSTANT_COLOR_EXT, GL_ONE_MINUS_SRC_COLOR);
    //glEnable(GL_BLEND);

    ///glEnable(GL_COLOR_MATERIAL);
    //glBlendFunc(GL_CONSTANT_COLOR_EXT, GL_ONE_MINUS_SRC_COLOR);
}

static void requestGL32()
{
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR, 3);
    glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 2);
    glfwOpenWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwOpenWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    glfwOpenWindowHint(GLFW_FSAA_SAMPLES, 4);

  #ifndef NDEBUG
    glfwOpenWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
  #endif
}

static GLFWwindow nbglPrepareWindow(bool fullscreen)
{
    GLFWvidmode vidMode;
    glfwGetDesktopMode(&vidMode);
    int winMode = fullscreen ? GLFW_FULLSCREEN : GLFW_WINDOWED;

    int width = vidMode.width / 2;
    int height = vidMode.height / 2;

    requestGL32();
    return glfwOpenWindow(width, height, winMode, "Milkyway@Home N-body", NULL);
}

void NBodyGraphics::display()
{
    const glm::mat4 modelMatrix = viewPole.CalcMatrix();
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClearDepth(1.0);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    if (this->galaxyModel)
    {
        this->galaxyModel->draw(modelMatrix);
    }

    if (this->drawOptions.drawAxes)
    {
        this->axes.draw(modelMatrix);
    }

    if (this->drawOptions.drawParticles)
    {
        this->drawBodies(modelMatrix);
    }

    if (this->drawOptions.drawInfo)
    {
        this->text.drawProgressText(this->scene);
    }
}

void NBodyGraphics::mainLoop()
{
    this->running = true;

    while (this->running)
    {
        double t1 = glfwGetTime();

        glfwPollEvents();
        if (!this->drawOptions.paused)
        {
            this->readSceneData();
        }
        this->display();
        glfwSwapBuffers();

        double t2 = glfwGetTime();

        double dt = t2 - t1;
        dt = dt - (1.0 / 60.0);
        if ((int) dt > 0)
        {
            mwMilliSleep((int) dt);
        }
    }
}

static void printVersionAndExts()
{
    std::cout << "GL version: "      << glGetString(GL_VERSION)
              << " Shader version: " << glGetString(GL_SHADING_LANGUAGE_VERSION)
              << std::endl;

    GLint n;
    glGetIntegerv(GL_NUM_EXTENSIONS, &n);
    for (GLint i = 0; i < n; i++)
    {
        printf("%s\n", glGetStringi(GL_EXTENSIONS, i));
    }
}

#if !BOINC_APPLICATION

scene_t* nbConnectSharedScene(int instanceId)
{
    int shmId;
    const int mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;
    struct stat sb;
    char name[128];
    scene_t* scene = NULL;

    if (snprintf(name, sizeof(name), "/milkyway_nbody_%d", instanceId) == sizeof(name))
    {
        mw_panic("name buffer too small for shared memory name\n");
    }

    shmId = shm_open(name, O_RDWR, mode);
    if (shmId < 0)
    {
        mwPerror("Error getting shared memory");
        return NULL;
    }

    if (fstat(shmId, &sb) < 0)
    {
        mwPerror("shmem fstat");
        shm_unlink(name);
        return NULL;
    }

    scene = (scene_t*) mmap(NULL, sb.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, shmId, 0);
    if (scene == MAP_FAILED)
    {
        mwPerror("mmap: Failed to mmap shared memory");
        if (shm_unlink(name) < 0)
        {
            mwPerror("Unlink shared memory");
        }

        return NULL;
    }

    if (   sb.st_size < sizeof(scene_t)
         || sb.st_size < (sizeof(scene_t) + 2 * scene->nbody * sizeof(FloatPos)))
    {
        mw_printf("Shared memory segment is impossibly small ("ZU")\n", (size_t) sb.st_size);
        if (shm_unlink(name) < 0)
        {
            mwPerror("Unlink shared memory");
        }

        return NULL;
    }

    return scene;
}

#else

static scene_t* nbAttemptConnectSharedScene(void)
{
    scene_t* scene = (scene_t*) mw_graphics_get_shmem(NBODY_BIN_NAME);
    if (!scene)
    {
        mw_printf("Failed to connect to shared scene\n");
    }

    return scene;
}

#define MAX_TRIES 5
#define RETRY_INTERVAL 250

/* In case the main application isn't ready yet, try and wait for a while */
scene_t* nbConnectSharedScene(int instanceId)
{
    int tries = 0;

    while (tries < MAX_TRIES)
    {
        if ((scene = nbAttemptConnectSharedScene()))
        {
            return alreadyAttached(); /* Error if something already attached */
        }

        mwMilliSleep(RETRY_INTERVAL);
        ++tries;
    }

    mw_printf("Could not attach to simulation after %d attempts\n", MAX_TRIES);
    return NULL;
}

#endif /* !BOINC_APPLICATION */

static double derivDiskShapeFunction(double r)
{
    return (1.0/4.0) * 4.0 * exp((-1.0/4.0) * std::fabs(r));
}

static double diskShapeFunction(double r)
{
    return 4.0 * exp((-1.0/4.0) * std::fabs(r));
}

void GalaxyModel::makePoint(NBodyVertex& point, bool neg, double r, double theta)
{
    double z;

    point.x = std::fabs(r) * cos(theta);
    point.y = std::fabs(r) * sin(theta);

    //if (std::fabs(r) < this->bulgeScale)
    if (false)
    {
        z = sqrt(sqr(this->bulgeScale) - 2.0 * sqr(r));
    }
    else
    {
        z = diskShapeFunction(r);
    }

    point.z = neg ? -z : z;
}

// generate top or bottom half of galaxy model
void GalaxyModel::generateSegment(bool neg)
{
    for (GLint i = this->radialSlices; i >= -this->radialSlices; --i)
    {
        double r = i * this->diameterSlice;
        double r1 = (i + 1) * this->diameterSlice;
        double theta;

        // wrap around an additional point to close the circle at 0
        for (GLint j = 0; j < this->axialSlices + 1; ++j)
        {
            theta = this->axialSliceSize * (double) j;

            makePoint(this->points[this->count++], neg, r, theta);
            makePoint(this->points[this->count++], neg, r1, theta);
        }
    }
}

// create section that tapers to 0 at the edge to join the upper and lower half
void GalaxyModel::generateJoiningSegment()
{
    double r = this->radialSlices * this->diameterSlice;

    // approximate slope of last segment at tip and continue a bit further
    double slope = derivDiskShapeFunction(r);
    double r1 = 1.1 * (slope * r + r);

    for (GLint j = 0; j < this->axialSlices + 1; ++j)
    {
        double theta = this->axialSliceSize * (double) j;

        makePoint(this->points[this->count++], true, r, theta);
        this->points[this->count++] = NBodyVertex(r1 * cos(theta), r1 * sin(theta), 0.0f);
    }

    for (GLint j = 0; j < this->axialSlices + 1; ++j)
    {
        double theta = this->axialSliceSize * (double) j;

        this->points[this->count++] = NBodyVertex(r1 * cos(theta), r1 * sin(theta), 0.0f);
        makePoint(this->points[this->count++], false, r, theta);
    }
}

void GalaxyModel::generateModel()
{
    GLuint segmentPoints = 2 * (2 * this->radialSlices + 1) * (this->axialSlices + 1);
    GLuint joinPoints = 2 * (this->axialSlices + 1);
    this->nPoints = 2 * (segmentPoints + joinPoints);

    this->points = new NBodyVertex[this->nPoints];
    this->colors = new Color[this->nPoints];
    this->count = 0;

    generateSegment(true);
    generateJoiningSegment();
    generateSegment(false);

#if 0
    for (GLint i = 0; i < nPoints; ++i)
    {
        printf("%f %f %f\n",
               this->points[i].x,
               this->points[i].y,
               this->points[i].z
            );
    }
#endif

    assert(this->count == this->nPoints);
}

int nbCheckConnectedVersion(const scene_t* scene)
{
    if (   scene->nbodyMajorVersion != NBODY_VERSION_MAJOR
        || scene->nbodyMinorVersion != NBODY_VERSION_MINOR)
    {
        mw_printf("Graphics version (%d.%d) does not match application version (%d.%d)\n",
                  NBODY_VERSION_MAJOR,
                  NBODY_VERSION_MINOR,
                  scene->nbodyMajorVersion,
                  scene->nbodyMinorVersion);
        return 1;
    }

    return 0;
}

int nbRunGraphics(const scene_t* scene, const VisArgs* args)
{
    if (!scene)
    {
        mw_printf("No scene to display\n");
        return 1;
    }

    if (!glfwInit())
    {
        mw_printf("Failed to initialize GLFW\n");
        return 1;
    }

    GLFWwindow window = nbglPrepareWindow(false);
    if (!window)
    {
        mw_printf("Failed to open window: %s\n", glfwErrorString(glfwGetError()));
        return 1;
    }


    // GL context needs to be open or else destructors will crash
    NBodyGraphics graphicsContext(scene);

    try
    {
        graphicsContext.prepareContext();
        nbglSetHandlers(&graphicsContext);
        graphicsContext.populateBuffers();
        graphicsContext.loadColors();

        GalaxyModel model;
        graphicsContext.loadModel(model);
        graphicsContext.mainLoop();
    }
    catch (const std::exception& e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
        glfwTerminate();
        return 1;
    }

    globalGraphicsContext = NULL;
    glfwTerminate();

    return 0;
}

