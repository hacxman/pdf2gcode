#!/usr/bin/python
import os
import sys
from PIL import Image
#import png
import numpy
import itertools
# port to pypng

if __name__=="__main__":
  print 'Converting to png'
  r = os.system('pdftoppm -r 600 -png {} > {}_1.png'.format(sys.argv[1], os.path.basename(sys.argv[1])))
  if r != 0:
    print 'fail. Return code {}'.format(r)
  im = Image.open('{}_1.png'.format(os.path.basename(sys.argv[1])))
  print 'Searching for crop'
  #im = png.Reader(file=open('{}_1.png'.format(sys.argv[1])))
#  imageW, imageH, pixels, meta = im.asDirect() #im.read()
#  print meta
#  pixels = list(pixels)
#  print pixels
#  print len(pixels)
#  image_2d = numpy.vstack(itertools.imap(numpy.uint8, pixels))
#  image_3d = numpy.reshape(image_2d, (imageW, imageH, meta['planes']))
#  print image_3d
  
  imageW = im.size[0]
  imageH = im.size[1]
   
  minX, maxX = imageW, 0
  minY, maxY = imageH, 0

  pixels = list(im.getdata())
#  if (im.mode == "RGB"):
  done = False
  for y in range(0, imageH):
    for x in range(0, imageW):
      offset = y*imageW + x
      rgb = pixels[offset]
      #rgb = image_3d[x, y]
#      print rgb
      if rgb[0] == 0 and rgb[1] == 0 and rgb[2] == 0:
          minX = min(minX, x)
          minY = min(minY, y)
          maxX = max(maxX, x)
          maxY = max(maxY, y)
  print minX, minY, maxX, maxY
  del pixels
  del im

  import subprocess
  print 'Cropping'
  print(        'pdftoppm -r 600 -x {} -y {} -W {} -H {} -png {} > {}.png'.
      format(minX-1, minY-1, (maxX-minX)+3, (maxY-minY)+3, sys.argv[1], os.path.basename(sys.argv[1])))
  #with open(os.path.basename(sys.argv[1])+'png', 'w') as fout:
  #    subprocess.call(['pdftoppm', '-r 600 -x {} -y {} -W {} -H {} -png {}'.
  #        format(minX-1, minY-1, (maxX-minX)+3, (maxY-minY)+3, sys.argv[1])], stdout=fout)

  r = os.system('pdftoppm -r 600 -x {} -y {} -W {} -H {} -png {} > {}.png'.
      format(minX-1, minY-1, (maxX-minX)+3, (maxY-minY)+3, sys.argv[1], os.path.basename(sys.argv[1])))
  if r != 0:
    print 'fail. Return code {}'.format(r)

  print 'Tracing'
  fname = os.path.basename(sys.argv[1])
  r = os.system('tracer {}.png > {}.ngc'.format(fname, fname))
  if r != 0:
    print 'fail. Return code {}'.format(r)
  print 'Created {}.ngc'.format(fname)
  print 25.4*((maxX-minX)+3)/600, 25.4*((maxY-minY)+3)/600
  import string
  with open('Gridprobe.txt') as fin:
      tpl = string.Template(fin.read())
      out = tpl.substitute(XSTEP=(25.4*((maxX-minX)+3)/600.0+2)/10.0, YSTEP=(25.4*((maxY-minY)+3)/600.0+2)/10.0,
              XSTART=-1, YSTART=-1)
      with open('{}.levelling.ngc'.format(fname), 'w') as fout:
          fout.write(out)
      print 'Generated {}.levelling.ngc'.format(fname)


