#*------------------------------------------------------------------------------*
#* JAX-FLUIDS -                                                                 *
#*                                                                              *
#* A fully-differentiable CFD solver for compressible two-phase flows.          *
#* Copyright (C) 2022  Deniz A. Bezgin, Aaron B. Buhendwa, Nikolaus A. Adams    *
#*                                                                              *
#* This program is free software: you can redistribute it and/or modify         *
#* it under the terms of the GNU General Public License as published by         *
#* the Free Software Foundation, either version 3 of the License, or            *
#* (at your option) any later version.                                          *
#*                                                                              *
#* This program is distributed in the hope that it will be useful,              *
#* but WITHOUT ANY WARRANTY; without even the implied warranty of               *
#* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                *
#* GNU General Public License for more details.                                 *
#*                                                                              *
#* You should have received a copy of the GNU General Public License            *
#* along with this program.  If not, see <https://www.gnu.org/licenses/>.       *
#*                                                                              *
#*------------------------------------------------------------------------------*
#*                                                                              *
#* CONTACT                                                                      *
#*                                                                              *
#* deniz.bezgin@tum.de // aaron.buhendwa@tum.de // nikolaus.adams@tum.de        *
#*                                                                              *
#*------------------------------------------------------------------------------*
#*                                                                              *
#* Munich, April 15th, 2022                                                     *
#*                                                                              *
#*------------------------------------------------------------------------------*

from abc import ABC, abstractmethod

from jaxfluids.domain_information import DomainInformation

class ShockSensor(ABC):
    """Abstract Class for shock sensors. Shock sensors indicate 
    the presence of discontinuities by a marker.
    """

    def __init__(self, domain_information: DomainInformation) -> None:
        self.domain_information = domain_information
        self.cell_sizes         = self.domain_information.cell_sizes
        self.active_axis_indices    = [{"x": 0, "y": 1, "z": 2}[axis] for axis in domain_information.active_axis]
    
    @abstractmethod
    def compute_sensor_function(self):
        """Computes the sensor function which is a marker (0/1)
        indicating the presence of shock discontinuities.

        Implementation in child classes.
        """
        pass