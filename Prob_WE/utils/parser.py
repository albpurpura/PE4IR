"""
    Author: Marco Maggipinto
    Copyright: (C) 2019-2020 <http://www.dei.unipd.it/ 
    Department of Information Engineering> (DEI), <http://www.unipd.it/ University of Padua>, Italy
    License: <http://www.apache.org/licenses/LICENSE-2.0 Apache License, Version 2.0>
"""

import argparse


class Parser:

    def __init__(self, description):
        self.parser = argparse.ArgumentParser(description=description)

    def add_arguments(self, arguments):
        for arg in arguments:
            self.parser.add_argument(*arg[0], **arg[1])

    def get_dictionary(self):
        return vars(self.parser.parse_args())