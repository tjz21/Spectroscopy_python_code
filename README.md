# MolSpeckPy: A python code for simulating linear and nonlinear spectroscopy experiments

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

# References
The module computing Franck-Condon spectra for a GBOM is based on the algorithm outlined in 
B. de Souza, F. Neese, and R. Izsak, J. Chem. Phys. 148, 034104 (2018).
An extension of the formalism to nonlinear spectroscopy was developed recently by us and is outlined in
L. Allan, and T. J. Zuehlsdorff, J. Chem. Theory Comput. 21, 5625-5621 (2025)

The cumulant solutions of the GBOM, as well as the third order cumulant approximation as applied to systems sampled
from MD trajectories are detailed in 
T. J. Zuehlsdorff, A. Montoya-Castillo, J. A. Napoli, T. E. Markland, and C. M. Isborn, J. Chem. Phys. 151, 074111 (2019); 
L. Allan, and T. J. Zuehlsdorff, J. Chem. Phys. 160, 074108 (2024)
for linear spectroscopy and 
T. J. Zuehlsdorff, H. Hong, L. Shi, and C. M. Isborn, J. Chem. Phys. 153, 044127 (2020)
for nonlinear spectroscopy. An extension of the Cumulant formalism to account for non-Condon effects through dipole moment
fluctuations was developed in 
Z. R. Wiethorn, K. E. Hunter, T. J. Zuehlsdorff, and A.-M. Castillo, J. Chem. Phys. 59, 244114 (2023).

The theoretical underpinnings of hybrid approaches such as the E-ZTFC method are described in the following 
references:
T. J. Zuehlsdorff, and C. M. Isborn, J. Chem. Phys. 148, 024110 (2018), 
T. J. Zuehlsdorff, J. A. Napoli, J. M. Milanese, T. E. Markland, and C. M. Isborn, J. Chem. Phys. 149, 024107 (2018), 
T. J. Zuehlsdorff, A. Montoya-Castillo, J. A. Napoli, T. E. Markland, and C. M. Isborn, J. Chem. Phys. 151, 074111 (2019).

Parts of the interface allowing for the construction of the GBOM from TeraChem (http://www.petachem.com/products.html) 
output files are based on code written by Ajay Khanna. The implementation of "tamed" third order cumulant contributions, 
as well as a formultion of nonlinear spectroscopy for GBOM Hamiltonians was carried out by Lucas Allan. 

# License
Copyright (C) 2019-2025 Tim J. Zuehlsdorff (zuehlsdt@oregonstate.edu)

The source code is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not 
distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the 
implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the Mozilla Public License, v. 2.0, 
for more details.
