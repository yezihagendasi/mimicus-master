#!/usr/bin/env python
#coding=utf-8
'''
Copyright 2014 Nedim Srndic, University of Tuebingen

This file is part of Mimicus.

Mimicus is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Mimicus is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Mimicus.  If not, see <http://www.gnu.org/licenses/>.
##############################################################################
F.py

Reproduction of scenario F.

Created on March 5, 2014.
'''
# Mimicus是自由软件：您可以根据自由软件基金会发布的GNU通用公共许可证的条款对其进行重新分发和/或修改，或者许可证的第3版，或者（在您的选项中）任何其他版本。
# Mimicus的发行是希望它会有用，但没有任何保证；甚至没有对适销性或特定用途适用性的默示保证。有关更多详细信息，请参阅GNU通用公共许可证。

from argparse import ArgumentParser
import sys
from common import attack_gdkde, attack_mimicry

def main():
    # Parse command-line arguments

    parser = ArgumentParser()
    parser.add_argument('--plot', help='Where to save plot (file name)',
                        default=False)
    args = parser.parse_args()
    
    # Perform the attacks
    scenario_name = 'F'
    attack_mimicry(scenario_name, args.plot)
    attack_gdkde(scenario_name, args.plot)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
