/*****************************************************************************
 *                                                                           *
 *  Copyright (C) 2010 Shane Reilly, Heidi Newberg, Malik Magdon-Ismail,     *
 *  Carlos Varela, Boleslaw Szymanski, and Rensselaer Polytechnic Institute  *
 *                                                                           *
 *  This file is part of the Light Modeling Library (LModL).                 *
 *                                                                           *
 *  This library is free software: you can redistribute it and/or modify     *
 *  it under the terms of the GNU General Public License as published by     *
 *  the Free Software Foundation, either version 3 of the License, or        *
 *  (at your option) any later version.                                      *
 *                                                                           *
 *  This library is distributed in the hope that it will be useful,          *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of           *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the             *
 *  GNU General Public License for more details.                             *
 *                                                                           *
 *  You should have received a copy of the GNU General Public License        *
 *  along with this library. If not, see <http://www.gnu.org/licenses/>.     *
 *                                                                           *
 *  Shane Reilly                                                             *
 *  reills2@cs.rpi.edu                                                       *
 *                                                                           *
 *****************************************************************************/

#include "trigtbl.hpp"

const float TRIG_TABLE[TRIGTBL_SIZE] =
{
    0.0000000000000000e+00,
    6.2790519529313374e-02,
    1.2533323356430426e-01,
    1.8738131458572463e-01,
    2.4868988716485479e-01,
    3.0901699437494740e-01,
    3.6812455268467797e-01,
    4.2577929156507272e-01,
    4.8175367410171532e-01,
    5.3582679497899666e-01,
    5.8778525229247314e-01,
    6.3742398974868975e-01,
    6.8454710592868873e-01,
    7.2896862742141155e-01,
    7.7051324277578925e-01,
    8.0901699437494745e-01,
    8.4432792550201508e-01,
    8.7630668004386369e-01,
    9.0482705246601958e-01,
    9.2977648588825146e-01,
    9.5105651629515353e-01,
    9.6858316112863108e-01,
    9.8228725072868872e-01,
    9.9211470131447788e-01,
    9.9802672842827156e-01,
    1.0000000000000000e+00,
    9.9802672842827156e-01,
    9.9211470131447776e-01,
    9.8228725072868861e-01,
    9.6858316112863108e-01,
    9.5105651629515353e-01,
    9.2977648588825135e-01,
    9.0482705246601947e-01,
    8.7630668004386347e-01,
    8.4432792550201496e-01,
    8.0901699437494745e-01,
    7.7051324277578925e-01,
    7.2896862742141144e-01,
    6.8454710592868850e-01,
    6.3742398974868952e-01,
    5.8778525229247325e-01,
    5.3582679497899666e-01,
    4.8175367410171521e-01,
    4.2577929156507249e-01,
    3.6812455268467775e-01,
    3.0901699437494712e-01,
    2.4868988716485482e-01,
    1.8738131458572457e-01,
    1.2533323356430409e-01,
    6.2790519529313138e-02,
    -3.2162857446782489e-16,
    -6.2790519529313346e-02,
    -1.2533323356430429e-01,
    -1.8738131458572477e-01,
    -2.4868988716485502e-01,
    -3.0901699437494773e-01,
    -3.6812455268467831e-01,
    -4.2577929156507266e-01,
    -4.8175367410171538e-01,
    -5.3582679497899677e-01,
    -5.8778525229247336e-01,
    -6.3742398974868997e-01,
    -6.8454710592868873e-01,
    -7.2896862742141155e-01,
    -7.7051324277578936e-01,
    -8.0901699437494734e-01,
    -8.4432792550201530e-01,
    -8.7630668004386358e-01,
    -9.0482705246601980e-01,
    -9.2977648588825146e-01,
    -9.5105651629515353e-01,
    -9.6858316112863119e-01,
    -9.8228725072868872e-01,
    -9.9211470131447788e-01,
    -9.9802672842827156e-01,
    -1.0000000000000000e+00,
    -9.9802672842827156e-01,
    -9.9211470131447788e-01,
    -9.8228725072868861e-01,
    -9.6858316112863108e-01,
    -9.5105651629515364e-01,
    -9.2977648588825124e-01,
    -9.0482705246601958e-01,
    -8.7630668004386336e-01,
    -8.4432792550201496e-01,
    -8.0901699437494701e-01,
    -7.7051324277578903e-01,
    -7.2896862742141155e-01,
    -6.8454710592868828e-01,
    -6.3742398974868963e-01,
    -5.8778525229247258e-01,
    -5.3582679497899632e-01,
    -4.8175367410171532e-01,
    -4.2577929156507222e-01,
    -3.6812455268467786e-01,
    -3.0901699437494679e-01,
    -2.4868988716485449e-01,
    -1.8738131458572468e-01,
    -1.2533323356430379e-01,
    -6.2790519529313263e-02,
    6.4325714893564978e-16,
    6.2790519529313665e-02,
    1.2533323356430418e-01,
    1.8738131458572507e-01,
    2.4868988716485488e-01,
    3.0901699437494801e-01,
    3.6812455268467820e-01,
    4.2577929156507255e-01,
    4.8175367410171566e-01,
    5.3582679497899666e-01,
    5.8778525229247358e-01,
    6.3742398974868986e-01,
    6.8454710592868928e-01,
    7.2896862742141177e-01,
    7.7051324277578925e-01,
    8.0901699437494778e-01,
    8.4432792550201519e-01,
    8.7630668004386392e-01,
    9.0482705246601969e-01,
    9.2977648588825146e-01,
    9.5105651629515375e-01,
    9.6858316112863119e-01,
    9.8228725072868883e-01,
    9.9211470131447788e-01,
    9.9802672842827156e-01,
    1.0000000000000000e+00,
};

