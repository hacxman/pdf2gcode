#!/usr/bin/python

import os
import sys

W, H = int(sys.argv[1]), int(sys.argv[2])
fname = sys.argv[3]

if __name__ == '__main__':
    with open('probe-results.txt') as fin:
      with open('probe-results.ssv', 'w') as fout:
        px, py = 0, 0
        lineresults = []
        for line in fin.readlines():
            print line
            x,y,z = map(float, line.split()[:3])
            fout.write(' '.join(map(str,[x,y,z]))+'\n')
#            fout.write('{} {} {}, '.format(x,y,z))
            #lineresults.append((x,y,z))
            #px += 1
            #if px >= W:
            #    px = 0
            #    py += 1
            #    if py % 2 == 0:
            #        lineresults.reverse()
            #    fout.write(', '.join(map(lambda (_x,_y,_z): '{} {} {}'.format(_x, _y, _z), lineresults))+'\n')
            #    lineresults = []
            #if py >= H:
            #    break

with open(fname) as fin:
    linecount = len(fin.readlines())
r=os.system('./scale_gcode.py {} "1-{}" --zlevel probe-results.ssv 0.05 | tee {}.levelled.ngc'.  format(fname, linecount, fname))
if r != 0:
    print 'levelling failed {}'.format(r)
