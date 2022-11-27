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

import logging
import os 
import sys
from typing import Dict, List

class Logger:
    """Logger for the JAX-FLUIDS solver.
    Logs information during the simulation to file and/or screen."""

    def __init__(self, logger_name: str = "", logging_level: str = "INFO") -> None:
        self.logger_name = logger_name
        
        self.level_dict    = {
            "DEBUG": logging.DEBUG, 
            "INFO": logging.INFO, 
            "WARNING": logging.WARNING, 
            "ERROR": logging.ERROR, 
            "NONE": logging.CRITICAL}

        self.is_streamoutput = True
        if logging_level in ["DEBUG_TO_FILE", "INFO_TO_FILE"]:
            logging_level = logging_level[:-8]
            self.is_streamoutput = False

        self.logging_level = self.level_dict[logging_level]

    def configure_logger(self, log_path: str) -> None:
        """Configures the logger. Sets up formatter, file and 
        stream handler. 

        :param log_path: Path to which logs are saved.
        :type log_path: str
        """
        logger = logging.getLogger(self.logger_name)
        if logger.hasHandlers():
            logger.handlers.clear()

        logger.setLevel(self.logging_level)
        formatter = logging.Formatter('%(message)s')

        if self.is_streamoutput:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(self.logging_level)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(os.path.join(os.path.abspath(log_path), 'output.log'))
        file_handler.setLevel(self.logging_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        self.logger = logger

    def log_jax_fluids(self) -> None:
        self.logger.info("*                                                                              *")
        self.logger.info("*          _     _    __  __        _____  _      _   _  ___  ____   ____      *")
        self.logger.info("*         | |   / \   \ \/ /       |  ___|| |    | | | ||_ _||  _ \ / ___|     *")
        self.logger.info("*      _  | |  / _ \   \  /  _____ | |_   | |    | | | | | | | | | |\___ \     *")
        self.logger.info("*     | |_| | / ___ \  /  \ |_____||  _|  | |___ | |_| | | | | |_| | ___) |    *")
        self.logger.info("*      \___/ /_/   \_\/_/\_\       |_|    |_____| \___/ |___||____/ |____/     *")
        self.logger.info("*                                                                              *")

    def log_copyright(self) -> None:
        self.logger.info("*------------------------------------------------------------------------------*")
        self.logger.info("* JAX-FLUIDS -                                                                 *")
        self.logger.info("*                                                                              *")
        self.logger.info("* A fully-differentiable CFD solver for compressible two-phase flows.          *") 
        self.logger.info("* Copyright (C) 2022  Deniz A. Bezgin, Aaron B. Buhendwa, Nikolaus A. Adams    *")
        self.logger.info("*                                                                              *")
        self.logger.info("* This program is free software: you can redistribute it and/or modify         *")
        self.logger.info("* it under the terms of the GNU General Public License as published by         *")
        self.logger.info("* the Free Software Foundation, either version 3 of the License, or            *")
        self.logger.info("* (at your option) any later version.                                          *")
        self.logger.info("*                                                                              *")
        self.logger.info("* This program is distributed in the hope that it will be useful,              *")
        self.logger.info("* but WITHOUT ANY WARRANTY; without even the implied warranty of               *")
        self.logger.info("* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                *")
        self.logger.info("* GNU General Public License for more details.                                 *")
        self.logger.info("*                                                                              *")
        self.logger.info("* You should have received a copy of the GNU General Public License            *")
        self.logger.info("* along with this program.  If not, see <https://www.gnu.org/licenses/>.       *")
        self.logger.info("*                                                                              *")
        self.logger.info("*------------------------------------------------------------------------------*")
        self.logger.info("*                                                                              *")
        self.logger.info("* CONTACT                                                                      *")
        self.logger.info("*                                                                              *")
        self.logger.info("* deniz.bezgin@tum.de // aaron.buhendwa@tum.de // nikolaus.adams@tum.de        *")               
        self.logger.info("*                                                                              *")
        self.logger.info("*------------------------------------------------------------------------------*")
        self.logger.info("*                                                                              *")
        self.logger.info("* Munich, April 15th, 2022                                                     *")
        self.logger.info("*                                                                              *")
        self.logger.info("*------------------------------------------------------------------------------*")

    def log_initialization(self) -> None:
        """Logs the initialization of the SimManager.
        """
        self.hline()
        self.nline()
        self.log_jax_fluids()
        self.logger.info("{}{:^78}{}".format("*","By BB - ML@AER","*"))
        self.nline()
        self.log_copyright()
        self.nline()

    def log_sim_start(self) -> None:
        """Logs the simulation start.
        """
        self.hline()
        self.nline()
        self.log_jax_fluids()
        self.nline()
        self.hline()

    def log_sim_finish(self, end_time: float) -> None:
        """Logs the simulation end.

        :param end_time: Final simulation time.
        :type end_time: float
        """
        self.hline()
        self.nline()
        self.logger.info("{}{:^78}{}".format("*","SIMULATION FINISHED SUCCESSFULLY","*"))
        self.logger.info("{}{:^78}{}".format("*","SIMULATION TIME %.3es" % end_time,"*"))
        self.log_jax_fluids()
        self.nline()
        self.hline()

        self._shutdown_logger()

    def _shutdown_logger(self) -> None:
        """Shutsdown logger. 
        Closes handlers and removes them from logger. 
        """
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
        
    def log_numerical_setup_and_case_setup(self, numerica_setup_dict: Dict, case_setup_dict: Dict) -> None:
        """Logs numerical setup and input file.

        :param numerica_setup_dict: Dictionary which contains information on the numerical setup.
        :type numerica_setup_dict: Dict
        :param case_setup_dict: Dictionary which contains information on the case setup.
        :type case_setup_dict: Dict
        """
        # LOG NUMERICAL SETUP
        self.nline()
        self.logger.info("{}{:^78}{}".format("*","NUMERICAL SETUP","*"))
        self.nline()
        for key0, item0 in numerica_setup_dict.items():
            self.logger.info("{}    {:<74}{}".format("*",key0,"*"))
            if isinstance(item0, dict):
                for key1, item1 in item0.items():
                    if isinstance(item1, dict):
                        self.logger.info("{}        {:<70}{}".format("*",key1,"*"))
                        for key2, item2 in item1.items():
                            self.logger.info("{}          {:<27}:    {:<36}{}".format("*",key2, str(item2),"*"))
                    else:
                        self.logger.info("{}        {:<29}:    {:<36}{}".format("*",key1, str(item1),"*")) 
            self.nline()      
        self.nline()
        self.hline()

        # LOG CASE SETUP
        self.nline()
        self.logger.info("{}{:^78}{}".format("*","CASE SETUP","*"))
        self.nline()
        for key0, item0 in case_setup_dict.items():
            self.logger.info("{}    {:<74}{}".format("*",key0,"*"))
            if isinstance(item0, dict):
                for key1, item1 in item0.items():
                    if isinstance(item1, dict):
                        self.logger.info("{}        {:<70}{}".format("*",key1,"*"))
                        for key2, item2 in item1.items():
                            self.logger.info("{}          {:<27}:    {:<36}{}".format("*",key2, str(item2),"*"))
                    else:
                        self.logger.info("{}        {:<29}:    {:<36}{}".format("*",key1, str(item1),"*")) 
            self.nline()      
        self.nline()
        
    def log_turbulent_stats_at_start(self, turb_stats_dict: Dict) -> None:
        """Logs the turbulent statistics of the initial turbulent flow field.

        :param turb_stats_dict: Dictionary which information on turbulent statistics.
        :type turb_stats_dict: Dict
        """
        self.nline()
        self.logger.info("{}{:^78}{}".format("*","INITIAL TURBULENT STATISTICS","*"))
        self.nline()
        for key, item in turb_stats_dict.items():
            self.logger.info("{}    {:<74}{}".format("*",str(key),"*"))
            for subkey, subitem in item.items():
                self.logger.info("{}        {:<25}:    {:<40}{}".format("*",str(subkey), str(subitem),"*"))       
            self.nline()
        self.nline()
        self.hline()
        
    def log_turbulent_stats_at_start(self, turb_stats_dict: Dict) -> None:
        """Logs the turbulent statistics of the initial turbulent flow field.

        :param turb_stats_dict: Dictionary which information on turbulent statistics.
        :type turb_stats_dict: Dict
        """
        self.nline()
        self.logger.info("{}{:^78}{}".format("*","INITIAL TURBULENT STATISTICS","*"))
        self.nline()
        for key, item in turb_stats_dict.items():
            self.logger.info("{}    {:<74}{}".format("*",str(key),"*"))
            for subkey, subitem in item.items():
                self.logger.info("{}        {:<25}:    {:<40}{}".format("*",str(subkey), str(subitem),"*"))       
            self.nline()
        self.nline()
        self.hline()

    def log_turbulent_stats_running(self, turb_stats_running_dict: Dict) -> None:
        """Logs the running turbulent statistics during a turbulent simulation.

        :param turb_stats_running_dict: Dictionary which information on turbulent statistics.
        :type turb_stats_running_dict: Dict
        """
        # TODO
        raise NotImplementedError

    def log_start_time_step(self, info_list: List) -> None:
        """Logs information at the beginning of an integration step.

        :param info_list: List of strings to be printed at the start
            of an integration step. 
        :type info_list: List
        """
        self.nline()
        for line in info_list:
            self.logger.info("{}    {:<74}{}".format("*", line, "*"))
        self.nline()

    def log_end_time_step(self, info_list: List) -> None:
        """Logs information at the end of an integration step.

        :param info_list: List of strings to be printed at the end
            of an integration step.
        :type info_list: List
        """
        self.nline()
        for line in info_list:
            self.logger.info("{}    {:<74}{}".format("*", line, "*"))
        self.nline()
        self.hline()
    
    def hline(self) -> None:
        """Inserts a dashed horizontal line in log.
        """
        self.logger.info("{}{}{}".format("*","-"*78,"*"))

    def nline(self) -> None:
        """Inserts a line break in log.
        """
        self.logger.info("{:<40}{:>40}".format("*","*"))