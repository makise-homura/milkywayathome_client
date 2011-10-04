# Copyright 2010 Matthew Arsenault, Travis Desell, Dave Przybylo,
# Nathan Cole, Boleslaw Szymanski, Heidi Newberg, Carlos Varela, Malik
# Magdon-Ismail and Rensselaer Polytechnic Institute.
#
# This file is part of Milkway@Home.
#
# Milkyway@Home is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Milkyway@Home is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Milkyway@Home.  If not, see <http://www.gnu.org/licenses/>.
#

include(CPUNameTest)
include(CheckIncludeFiles)

if(SYSTEM_IS_X86)
  if(NOT MSVC)
    set(BASE_SSE_FLAGS "-mfpmath=sse -msse")
    set(SSE2_FLAGS "${BASE_SSE_FLAGS} -msse2")
    set(SSE3_FLAGS "${SSE2_FLAGS} -msse3")
    set(SSE4_FLAGS "${SSE3_FLAGS} -msse4")
    set(SSE41_FLAGS "${SSE4_FLAGS} -msse4.1")
    set(AVX_FLAGS "${SSE41_FLAGS} -mavx")

    set(DISABLE_SSE2_FLAGS "-mno-sse2")
    set(DISABLE_SSE2_FLAGS "-mfpmath=387 -mno-sse ${DISABLE_SSE2_FLAGS}")
    set(DISABLE_SSE3_FLAGS "-mno-sse3")
    set(DISABLE_SSE41_FLAGS "-mno-sse4.1")
    set(DISABLE_AVX_FLAGS "-mno-avx")
  else()
    set(SSE2_FLAGS "/arch:SSE2 /D__SSE2__=1")
    set(DISABLE_SSE2_FLAGS "")
    set(DISABLE_SSE3_FLAGS "")
    set(DISABLE_SSE41_FLAGS "")
    set(DISABLE_AVX_FLAGS "")

    # MSVC doesn't generate SSE3 itself, and doesn't define this
    set(SSE3_FLAGS "${SSE2_FLAGS} /D__SSE3__=1")
    set(SSE41_FLAGS "${SSE3_FLAGS} /D__SSE4_1__=1")
    set(AVX_FLAGS "/arch:AVX /D__AVX__=1 ${SSE2_FLAGS}")
  endif()
endif()

set(CMAKE_REQUIRED_FLAGS "${AVX_FLAGS}")

# On OS X 10.6 the macports GCC can support AVX, but you must use the
# system assembler which doesn't and fails
try_compile(AVX_CHECK ${CMAKE_BINARY_DIR} ${MILKYWAYATHOME_CLIENT_CMAKE_MODULES}/test_avx.c
                CMAKE_FLAGS "-DCMAKE_C_FLAGS:STRING=${AVX_FLAGS}")
if(AVX_CHECK)
  message(STATUS "AVX compiler flags - '${AVX_FLAGS}'")
  set(HAVE_AVX TRUE CACHE INTERNAL "Compiler has AVX support")
endif()

mark_as_advanced(HAVE_AVX)



set(CMAKE_REQUIRED_FLAGS "${SSE41_FLAGS}")
check_include_files(smmintrin.h HAVE_SSE41 CACHE INTERNAL "Compiler has SSE4.1 headers")
mark_as_advanced(HAVE_SSE41)

set(CMAKE_REQUIRED_FLAGS "${SSE3_FLAGS}")
check_include_files(pmmintrin.h HAVE_SSE3 CACHE INTERNAL "Compiler has SSE3 headers")
mark_as_advanced(HAVE_SSE3)

set(CMAKE_REQUIRED_FLAGS "${SSE2_FLAGS}")
check_include_files(emmintrin.h HAVE_SSE2 CACHE INTERNAL "Compiler has SSE2 headers")
mark_as_advanced(HAVE_SSE2)

set(CMAKE_REQUIRED_FLAGS "")

if(APPLE AND SYSTEM_IS_X86)
  set(ALWAYS_HAVE_SSE2 TRUE CACHE INTERNAL "System always has SSE2")
elseif(SYSTEM_IS_X86_64)
  set(ALWAYS_HAVE_SSE2 TRUE CACHE INTERNAL "System always has SSE2")
else()
  set(ALWAYS_HAVE_SSE2 FALSE CACHE INTERNAL "System always has SSE2")
endif()

if(APPLE AND SYSTEM_IS_X86)
  set(ALWAYS_HAVE_SSE3 TRUE CACHE INTERNAL "System always has SSE3")
else()
  set(ALWAYS_HAVE_SSE3 FALSE CACHE INTERNAL "System always has SSE3")
endif()

function(disable_sse41 target)
  get_target_property(comp_flags ${target} COMPILE_FLAGS)
  if(comp_flags STREQUAL "comp_flags-NOTFOUND")
    set(comp_flags "")
  endif()
  set_target_properties(${target}
                          PROPERTIES
                            COMPILE_FLAGS "${comp_flags} ${DISABLE_SSE4_FLAGS}")
endfunction()

function(disable_sse3 target)
  get_target_property(comp_flags ${target} COMPILE_FLAGS)
  if(comp_flags STREQUAL "comp_flags-NOTFOUND")
    set(comp_flags "")
  endif()
  set_target_properties(${target}
                          PROPERTIES
                            COMPILE_FLAGS "${comp_flags} ${DISABLE_SSE3_FLAGS}")
endfunction()

function(disable_sse2 target)
  get_target_property(comp_flags ${target} COMPILE_FLAGS)
  if(comp_flags STREQUAL "comp_flags-NOTFOUND")
    set(comp_flags "")
  endif()
  set_target_properties(${target}
                          PROPERTIES
                            COMPILE_FLAGS "${comp_flags} ${DISABLE_SSE2_FLAGS}")
endfunction()

function(enable_sse41 target)
  get_target_property(comp_flags ${target} COMPILE_FLAGS)
  if(comp_flags STREQUAL "comp_flags-NOTFOUND")
    set(comp_flags "")
  endif()

  set_target_properties(${target}
                          PROPERTIES
                            COMPILE_FLAGS "${comp_flags} ${SSE41_FLAGS}")
  get_target_property(new_comp_flags ${target} COMPILE_FLAGS)
endfunction()

function(enable_sse3 target)
  get_target_property(comp_flags ${target} COMPILE_FLAGS)
  if(comp_flags STREQUAL "comp_flags-NOTFOUND")
    set(comp_flags "")
  endif()

  set_target_properties(${target}
                          PROPERTIES
                            COMPILE_FLAGS "${comp_flags} ${SSE3_FLAGS}")
  get_target_property(new_comp_flags ${target} COMPILE_FLAGS)
endfunction()

function(enable_sse2 target)
  get_target_property(comp_flags ${target} COMPILE_FLAGS)
  if(comp_flags STREQUAL "comp_flags-NOTFOUND")
    set(comp_flags "")
  endif()

  set_target_properties(${target}
                          PROPERTIES
                            COMPILE_FLAGS "${comp_flags} ${SSE2_FLAGS}")
endfunction()

function(enable_avx target)
  get_target_property(comp_flags ${target} COMPILE_FLAGS)
  if(comp_flags STREQUAL "comp_flags-NOTFOUND")
    set(comp_flags "")
  endif()

  set_target_properties(${target}
                          PROPERTIES
                            COMPILE_FLAGS "${comp_flags} ${AVX_FLAGS}")
  get_target_property(new_comp_flags ${target} COMPILE_FLAGS)
endfunction()


function(maybe_disable_ssen)
  if(SYSTEM_IS_X86)
    foreach(i ${ARGN})
      if(NOT ALWAYS_HAVE_SSE2)
        disable_sse2(${i})
      endif()
      if(NOT ALWAYS_HAVE_SSE3)
        disable_sse3(${i})
      endif()
    endforeach()
  endif()
endfunction()

