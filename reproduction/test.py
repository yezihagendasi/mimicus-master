#!/usr/bin/env python
# coding=utf-8
from argparse import ArgumentParser
import sys
# from common import  attack_mimicry
from common import attack_gdkde
# print 'aaa'
import config

# attack_gdkde('F')
print config.get('datasets', 'contagio')
# with open('I:\\mimicus\\mimicus-master\\data\\contagio.csv', 'wb+') as f:
#     print 'aaa'


# def main():
#     # Parse command-line arguments
#     # print 'csacaca'
#     # parser = ArgumentParser()
#     # print '------'
#     # parser.add_argument('--plot', help='Where to save plot (file name)',
#     #                     default=False)
#     # args = parser.parse_args()
#     # print 'aaaa'
#     # # Perform the attacks
#     # scenario_name = 'F'
#     # print 'bbbb'
#     # print scenario_name
#     # # print args.plot
#     # attack_mimicry(scenario_name, args.plot)
#     # # attack_gdkde(scenario_name, args.plot)
#     print 'aaaaa'
#
#     return 0
#
#
# if __name__ == '__main__':
#     # sys.exit(main())
#     main()
