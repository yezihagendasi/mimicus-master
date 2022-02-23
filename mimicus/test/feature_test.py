#coding=utf-8
from mimicus.tools import  featureedit
import os

import subprocess
import platform
win = platform.system() == "Windows"

# pwd = os.path.dirname(__file__)
#
#
# print pwd
# # tesseract_exe_name = os.path.join(pwd,'tesseract')
#
# pdf = featureedit.FeatureEdit('I:/mimicus/mimicus-master/data/pdfs/ben/02govbnd.pdf')
perl = subprocess.Popen(['perl', '-ln0777e', r'print sprintf("%d", @-[1]) while /[^\w\d](startxref)[^\w\d]/g', 'I:/mimicus/mimicus-master/data/pdfs/ben/02govbnd.pdf'], shell=True,encoding="utf-8",stdout=subprocess.PIPE)
