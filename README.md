# Spectroscopy python code

This code is designed to compute a linear and nonlinear optical spectra of condensed phase systems under a 
range of different approximations. The two basic ways to specify a system in the code is either by constructing
a generalized brownian oscillator model (GBOM) of the ground and excited state potential energy surface (PES),
or by inputting a list of electronic energy gaps calculated along a molecular dynamics (MD) trajectory. A 
GBOM can be constructed by approximating the ground and excited state potential energy surface as harmonic around
their respective minima and relating the ground and excited state normal modes through a Duschinsky rotation. 

For the GBOM, a range of approximations to the linear absorption and emission spectra are implemented, ranging from
Franck-Condon to second- and 3rd order cumulant approximation, as well as classical and quantum ensemble approaches 
and the hybrid E-ZTFC method. If the model is constructed from an MD trajectory input, only cumulant and ensemble 
spectra can be computed. 

For nonlinear spectroscpy, only transient absorption and 2DES are implemented so far, and regardless of whether
an GBOM or an MD trajectory is used as an input, nonlinear spectra can only be computed in the second and third 
order cumulant approach. 

# License
Copyright (C) 2019-2020 Tim J. Zuehlsdorff (zuehlsdt@oregonstate.edu)

The source code is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not 
distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the 
implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the Mozilla Public License, v. 2.0, 
for more details.
