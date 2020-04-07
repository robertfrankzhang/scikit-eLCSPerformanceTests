"""
Name:        eLCS_Run.py
Authors:     Ryan Urbanowicz - Written at Dartmouth College, Hanover, NH, USA
Contact:     ryan.j.urbanowicz@darmouth.edu
Created:     November 1, 2013
Description: To run e-LCS, run this module.  A properly formatted configuration file, including all run parameters must be included with the path to that 
             file given below.  In this example, the configuration file has been included locally, so only the file name is required.
             
---------------------------------------------------------------------------------------------------------------------------------------------------------
eLCS: Educational Learning Classifier System - A basic LCS coded for educational purposes.  This LCS algorithm uses supervised learning, and thus is most 
similar to "UCS", an LCS algorithm published by Ester Bernado-Mansilla and Josep Garrell-Guiu (2003) which in turn is based heavily on "XCS", an LCS 
algorithm published by Stewart Wilson (1995).  

Copyright (C) 2013 Ryan Urbanowicz 
This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the 
Free Software Foundation; either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABLILITY 
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, 
Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

#Import Required Modules------------------------------------
from eLCS_Timer import Timer
from eLCS_ParamParser import ParamParser
from eLCS_Offline_Environment import Offline_Environment
from eLCS_Algorithm import eLCS
from eLCS_Constants import *
#-----------------------------------------------------------

#Obtain all run parameters from the configuration file and store them in the 'Constants' module.
l = []
ParamParser("Datasets/Multiplexer6.csv", cv=3, labelPhenotype="class", learningIterations="10000", randomSeed=0)

for i in range(3):
    cons.setCV()
    timer = Timer()
    cons.referenceTimer(timer)
    env = Offline_Environment()
    cons.referenceEnv(env)
    cons.parseIterations()
    e = eLCS()
    print(e.testEval[0])
    l.append(e.testEval[0])
print(np.mean(np.array(l)))