/*
Copyright (C) 2011  Matthew Arsenault

This file is part of Milkway@Home.

Milkyway@Home is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Milkyway@Home is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Milkyway@Home.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef _LUA_INITIAL_CONDITIONS_H_
#define _LUA_INITIAL_CONDITIONS_H_

#include <lua.h>
#include "nbody_types.h"

InitialConditions* checkInitialConditions(lua_State* luaSt, int index);
int pushInitialConditions(lua_State* luaSt, const InitialConditions* h);
int registerInitialConditions(lua_State* luaSt);

#ifdef __cplusplus
}
#endif

#endif /* _LUA_INITIAL_CONDITIONS_H_ */
