# -*- coding: utf-8 -*-

import os
import re
import sys
import shelve
from pprint import pprint 
import MeCab

def isKanji(ws):
  return sum([not (ord(w) == 0x3005 or ord(w) == 0x3007 or ord(w) == 0x303b or (ord(w) >= 0x3400 and ord(w) <= 0x9FFF) or (ord(w) >= 0xF900 and ord(w) <= 0xFAFF) or (ord(w) >= 0x20000 and ord(w) <= 0x2FFFF)) for w in ws]) == 0



output = ""
for shelve_file in [m.group(0) for m in [re.compile('n\d+\w\w\.shelve').match(fn) for fn in list(os.walk("."))[0][2]] if m]:
  shel = shelve.open(shelve_file)
  for text in shel['novel_honbun_arr']:
    text = text.replace('<br/>','').replace(u'\u3000', '').replace(u'<ruby>','').replace(u'</ruby>','').replace(u'<rt>','').replace(u'</rt>','').replace(u'<rp>','').replace(u'</rp>','').replace(u'<rb>','').replace(u'</rb>','')
    for w in text:
      if isKanji(w) or w == "\n":
      	output += w
      else:
      	output += "　"

f = open('jyukugo.txt', 'w')
f.write(output)
f.close()