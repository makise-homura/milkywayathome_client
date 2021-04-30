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

#ifndef _NBODY_GL_UTIL_H_
#define _NBODY_GL_UTIL_H_

#include "nbody_gl_includes.h"

struct Color
{
    GLfloat r, g, b;
    GLfloat ignore;
};

GLuint nbglCreateProgram(const char* name,
                         const char* vertSrc,
                         const char* fragSrc,
                         GLint vertSrcLen,
                         GLint fragSrcLen);

void nbglPrintGLVersionAndExts();

#endif /* _NBODY_GL_UTIL_H_ */


