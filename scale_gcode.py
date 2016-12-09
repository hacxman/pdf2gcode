#!/usr/bin/python
import sys
import os
import png
import itertools


def Affine_Fit(from_pts, to_pts):
    """Fit an affine transformation to given point sets.
      More precisely: solve (least squares fit) matrix 'A'and 't' from
      'p ~= A*q+t', given vectors 'p' and 'q'.
      Works with arbitrary dimensional vectors (2d, 3d, 4d...).

      Written by Jarno Elonen <elonen@iki.fi> in 2007.
      Placed in Public Domain.

      Based on paper "Fitting affine and orthogonal transformations
      between two sets of points, by Helmuth Spath (2003)."""

    q = from_pts
    p = to_pts
    if len(q) != len(p) or len(q) < 1:
        print>>sys.stderr, "from_pts and to_pts must be of same size."
        return False

    dim = len(q[0])  # num of dimensions
    if len(q) < dim:
        print>>sys.stderr, "Too few points => under-determined system."
        return False

    # Make an empty (dim) x (dim+1) matrix and fill it
    c = [[0.0 for a in range(dim)] for i in range(dim + 1)]
    for j in range(dim):
        for k in range(dim + 1):
            for i in range(len(q)):
                qt = list(q[i]) + [1]
                c[k][j] += qt[k] * p[i][j]

    # Make an empty (dim+1) x (dim+1) matrix and fill it
    Q = [[0.0 for a in range(dim)] + [0] for i in range(dim + 1)]
    for qi in q:
        qt = list(qi) + [1]
        for i in range(dim + 1):
            for j in range(dim + 1):
                Q[i][j] += qt[i] * qt[j]

    # Ultra simple linear system solver. Replace this if you need speed.
    def gauss_jordan(m, eps=1.0 / (10 ** 10)):
        """Puts given matrix (2D array) into the Reduced Row Echelon Form.
           Returns True if successful, False if 'm' is singular.
           NOTE: make sure all the matrix items support fractions! Int matrix will NOT work!
           Written by Jarno Elonen in April 2005, released into Public Domain"""
        (h, w) = (len(m), len(m[0]))
        for y in range(0, h):
            maxrow = y
            for y2 in range(y + 1, h):    # Find max pivot
                if abs(m[y2][y]) > abs(m[maxrow][y]):
                    maxrow = y2
            (m[y], m[maxrow]) = (m[maxrow], m[y])
            if abs(m[y][y]) <= eps:     # Singular?
                return False
            for y2 in range(y + 1, h):    # Eliminate column y
                c = m[y2][y] / m[y][y]
                for x in range(y, w):
                    m[y2][x] -= m[y][x] * c
        for y in range(h - 1, 0 - 1, -1):  # Backsubstitute
            c = m[y][y]
            for y2 in range(0, y):
                for x in range(w - 1, y - 1, -1):
                    m[y2][x] -= m[y][x] * m[y2][y] / c
            m[y][y] /= c
            for x in range(h, w):       # Normalize row y
                m[y][x] /= c
        return True

    # Augement Q with c and solve Q * a' = c by Gauss-Jordan
    M = [Q[i] + c[i] for i in range(dim + 1)]
    if not gauss_jordan(M):
        print>>sys.stderr, "Error: singular matrix. Points are probably coplanar."
        return False

    # Make a result object
    class Transformation:
        """Result object that represents the transformation
           from affine fitter."""

        def To_Str(self):
            res = ""
            for j in range(dim):
                str = "x%d' = " % j
                for i in range(dim):
                    str += "x%d * %f + " % (i, M[i][j + dim + 1])
                str += "%f" % M[dim][j + dim + 1]
                res += str + "\n"
            return res

        def Transform(self, pt):
            res = [0.0 for a in range(dim)]
            for j in range(dim):
                for i in range(dim):
                    res[j] += pt[i] * M[i][j + dim + 1]
                res[j] += M[dim][j + dim + 1]
            return res
    return Transformation()

from bilinear import bilinear
from math import sqrt


def pt_norm2(pt0, pt1):
    r0 = (pt1.x - pt0.x) ** 2 + (pt1.y - pt0.y) ** 2
    return sqrt(r0)


def get_path_len(pts):
    return reduce(lambda (o, lng), p: (p, lng + pt_norm2(o, p)), pts[1:], (pts[0], 0))

# this is ultra slow :)


def dump_region_to_png(pts, fname):
    ps = map(lambda p: p.to_pair(), pts)
    scale = 255 / float(len(ps) - 1)
    scales = 10
    maximalx = reduce(lambda o, p: max(o, p[0]), ps, 0) * scales
    maximaly = reduce(lambda o, p: max(o, p[1]), ps, 0) * scales
    minimalx = reduce(lambda o, p: min(o, p[0]), ps, maximalx) * scales
    minimaly = reduce(lambda o, p: min(o, p[1]), ps, maximaly) * scales
    img = [[(0, 0, 0) for x in range(int(maximalx + minimalx + 2))]
           for y in range(int(maximaly + minimaly + 2))]
    for i, (x, y) in enumerate(ps):
        #    print>>sys.stderr, i, x, y, len(img), int(y*scales+1)
        #print>>sys.stderr, 255-int(i*scale)
        img[int(y * scales)][int(x * scales)] = (max(0, 255 -
                                                     int(i * scale)), min(255, int(i * scale)), 0)
        img[int(y * scales + 1)][int(x * scales)] = (max(0, 255 -
                                                         int(i * scale)), min(255, int(i * scale)), 0)
        img[int(y * scales - 1)][int(x * scales)] = (max(0, 255 -
                                                         int(i * scale)), min(255, int(i * scale)), 0)
        img[int(y * scales)][int(x * scales + 1)] = (max(0, 255 -
                                                         int(i * scale)), min(255, int(i * scale)), 0)
        img[int(y * scales)][int(x * scales - 1)] = (max(0, 255 -
                                                         int(i * scale)), min(255, int(i * scale)), 0)

    ln = len(img)
    for y in range(len(img)):
        print>>sys.stderr, fname, y, 'of', ln, '   \r',
        sys.stdout.flush()
        img[y] = reduce(lambda o, p: o + list(p), img[y], [])
    # print
    png.from_array(img, 'RGB').save(fname)


def str_to_range(s):
    r = map(int, s.split('-'))
    if len(r) == 0:
        raise Exception("Empty range")
    elif len(r) == 1:  # only one line
        return xrange(r[0] - 1, r[0])
    elif len(r) == 2:
        return xrange(r[0] - 1, r[1])
    else:
        raise Exception("More than two numbers in a range")


def parse_ranges(inp):
    return reduce(lambda x, y: x + y,
                  map(list, map(str_to_range, inp.split(','))))


def parse_ranges2(inp):
    return map(str_to_range, inp.split(','))


def indexP(f, it):
    return filter(lambda (i, d): f(d), enumerate(it))[0]


def find_scale_and_replace(line, char, scale):
    idx1 = line.find(char)

    comment_idx = line.find('(')
    if comment_idx > -1 and comment_idx <= idx1:
        return line

    if idx1 >= 0:
        # print indexP(lambda d: d.isalpha(), line[idx1:])
        where, what = indexP(lambda d: d.isalpha()
                             or d.isspace(), line[idx1 + 1:])
        #idx2 = line.find(" ", idx1)
        idx2 = where + idx1
        if idx2 < 0 and line[-1].isspace() and line[-2].isdigit():
            idx2 = len(line) - 1
        precission = idx2 - line.find(".", idx1) - 1
        precission = 5
        newX = round(float(line[idx1 + 1:idx2]) * scale, precission)
        line = line[:idx1] + char + str("%0.0{}f".format(precission)%newX) + line[idx2:]

    return line


def find_and_replace(line, char, nval):
    if nval is None:
        return line
    idx1 = line.find(char)

    comment_idx = line.find('(')
    if comment_idx > -1 and comment_idx <= idx1:
        return line

    if idx1 >= 0:
        # print indexP(lambda d: d.isalpha(), line[idx1:])
        where, what = indexP(lambda d: d.isalpha()
                             or d.isspace(), line[idx1 + 1:])
        #idx2 = line.find(" ", idx1)
        idx2 = where + idx1
        if idx2 < 0 and line[-1].isspace() and line[-2].isdigit():
            idx2 = len(line) - 1
        precission = idx2 - line.find(".", idx1) - 1
        precission = 5
        newX = round(nval, precission)
        #line = line[:idx1] + char + str(newX) + line[idx2:]
        line = line[:idx1] + char + str("%0.0{}f".format(precission)%newX) + line[idx2:]

    return line


def find_transform_and_replace(line, aff):
    pt = Pt(line)
    if pt.x is None or pt.y is None:
        return line
    t = aff.Transform(pt.to_pair())
    pt.x = t[0]
    pt.y = t[1]
    return repr(pt)


def find_zlevel_and_replace(line, M, base_z_level=None):
    import bilinear
    import copy
    pt = Pt(line)
    if pt.x is None or pt.y is None:
        return line
    if base_z_level is None:
        base_z_level = 0.0
    t = bilinear.bilinear(pt.x, pt.y, M)
    if pt.z is None:
        pt.add_z_before(t + base_z_level, 'F')
        # print repr(pp).strip()
        # print "G00 Z%.5f"%t
    else:
        pt.z += t
    return repr(pt)

XSEP = " X"
YSEP = " Y"
ZSEP = " Z"


class Pt(object):

    def find_var(self, line, char):
        idx1 = line.find(char)

        comment_idx = line.find('(')
        if comment_idx > -1 and comment_idx <= idx1:
            return None

        if idx1 >= 0:
            #print>>sys.stderr, "LOL", line[idx1+1:]
            where, what = indexP(lambda d: d.isalpha()
                                 or d.isspace(), line[idx1 + 1:] + " ")
            #idx2 = line.find(" ", idx1)
            idx2 = where + idx1 + 1
            if idx2 < 0 and line[-1].isspace() and line[-2].isdigit():
                idx2 = len(line) - 1
            precission = idx2 - line.find(".", idx1) + 1
            return [char, float(line[idx1 + 1:idx2]), precission]
            line = line[:idx1] + char + str("%0.0{}f".format(precission)%newX) + line[idx2:]

    def to_pair(self):
        return (self.x, self.y)

    def to_triple(self):
        return (self.x, self.y, self.z)

    def add_z_before(self, val, beforewhat):
        idx = self.str_.find(beforewhat)
        self.str_ = self.str_[:idx] + "Z%.5f " % val + self.str_[idx:]

    def __init__(self, str_):
        # print str_
        self.prefix = str_.split(" ")[0]
        self.str_ = str_
        self.x_ = self.find_var(str_, 'X')
        self.x = self.x_[1] if self.x_ else None
        self.y_ = self.find_var(str_, 'Y')
        self.y = self.y_[1] if self.y_ else None
        self.z_ = self.find_var(str_, 'Z')
        self.z = self.z_[1] if self.z_ else None
        if (len(self.prefix) > str_.find('Y')
            or len(self.prefix) > str_.find('X')
            or len(self.prefix) > str_.find('Z')
            ) and \
           (str_.find('Y') != -1
                or str_.find('X') != -1
                or str_.find('Z') != -1):
            self.prefix = ''

        #print>>sys.stderr, "PREFIX=", self.prefix

    def __repr__(self):
        l = find_and_replace(self.str_, 'X', self.x)
        l = find_and_replace(l, 'Z', self.z)
        return find_and_replace(l, 'Y', self.y)
        # return self.prefix + \
        #  XSEP + str(round(self.x, self.x_[2])) + \
        #  YSEP + str(round(self.y, self.y_[2]))

from math import sqrt


def pt_dist_from(pt0):
    def pt_dist(pt1, pt2):
        r0 = (pt1.x - pt0.x) ** 2 + (pt1.y - pt0.y) ** 2
        r1 = (pt2.x - pt0.x) ** 2 + (pt2.y - pt0.y) ** 2
        return cmp(r0, r1)

    return pt_dist


def sort_path(unsorted_, sorted_):
    # print unsorted_, sorted_
    if sorted_ is None:
        return "Nesortol som"
    p0 = sorted_[-1]
    u2 = sorted(unsorted_, cmp=pt_dist_from(p0))
    p1 = u2[0]
    if len(unsorted_) == 1:
        return sorted_ + unsorted_
    return sort_path(u2[1:], sorted_ + [p1])


def command_name(line):
    try:
        return line.strip().split()[0]
    except IndexError:
        return ""


def lerp(v0, v1, t):
    return (1 - t) * v0 + t * v1

if __name__ == '__main__':
    if '--help' in sys.argv:
        print """Usage:
      {} GCODEFILE ranges [scale_factor] [--sort|--affine|--merge S E|--zlevel file_w_probed_data.ssv step_in_mm] [3+3 2D points]

      n.b. this tools accepts drill files as well!

      --merge S E: sometimes you want to sort points in all G code groups, so --merge tries
      to group all lines between S and E. ex: "1-99" --merge G99 G80, creates one big group of
      commands occurring between G99 and G80 and discards those between, since those can't serve
      any purpose now.

      ex: ./a.py lala.ngc "7-19,33-58" 0.94 >newfile.ngc
      performs a simple scaling on selected lines.

      ex: ./a.py lala.ngc "7-19,33-58" --sort >newfile.ngc
      performs a shortest path first walk type sort on selected lines.
      your machine should then spend less time traveling.

      ex: ./a.py lala.ngc "7-19,33-58" --affine 1 1 1 2 2 2   4 4 6 6 8 4 >newfile.ngc
      performs a affine transform on selected lines.
      transformation is defined by 3 original points and 3 transformed points.

      ex: ./a.py lala.ngc "7-10,33-58" --zlevel data.ssv 0.05
      levels your Z heigh for constant milling depth. intended for processing PCB milling
      files with data from Z probe. it interpolates moves using bilinear transform.
      input data are space separated triples x, y, z in mm COLS and ROWS.
      lines are cut to pieces set by last parameter (0.05mm).
        ex: 0.01 0.02 0.0023, 0.48 0.024 0.0031
            0.019 0.51 0.0026, 0.50 0.498 0.0025
            so this is 2x2 data in triples x y z
            NOTE the , separator between triples!

            n.b. for now - tool presumes EXACT EVEN SPACING between points in grid
    """.format(sys.argv[0])
        exit(1)
    ranges = parse_ranges(sys.argv[2])
    sort_ = "--sort" in sys.argv
    merge_ = "--merge" in sys.argv
    affine_ = "--affine" in sys.argv
    zlevel_ = "--zlevel" in sys.argv

    retract_mm = 2.0

    with open(sys.argv[1], "r") as fin:
        lines = fin.readlines()
        last_line = 0
        # print lines[1]
        if lines[1][0:6] == ";DRILL":
            XSEP = "X"
            YSEP = "Y"
            # print 'DRILL FILE'
        if sort_:
            ranges = parse_ranges2(sys.argv[2])
            for i, rng in enumerate(ranges):
                if merge_:
                    m_p = sys.argv.index("--merge")
                    S = sys.argv[m_p + 1]
                    E = sys.argv[m_p + 2]
                    # print S, E
                    sections = filter(lambda __x: __x[0].strip() == S or __x[0].strip() == E, zip(
                        lines[rng[0]:rng[-1] + 1], range(rng[0], rng[-1] + 1)))
                    # print sections
                    assert sections[0][0].strip() == S
                    rng = xrange(sections[0][1] + 1, rng[-1])
                    rngs = [range(sections[idx_][1], sections[idx_ + 1][1] + 1)
                            for idx_ in range(1, len(sections) - 1, 2)]
                    for r in rngs:
                        for lnm in r:
                            lines[lnm] = '\n'
                    # print lines
                    # print "KOKOT"

                pts_orig = map(Pt, lines[rng[0]:rng[-1] + 1])
                pts = filter(lambda __x: __x.x is not None, pts_orig)
#        print pts
                dump_region_to_png(pts, 'un_region{}.png'.format(i))
                p = pts[1:]
#        print 'sort path'
                new_path = sort_path(p, [pts[0]])
                print>>sys.stderr, 'Path len diff:', get_path_len(
                    pts)[1] - get_path_len(new_path)[1]
                dump_region_to_png(new_path, 'region{}.png'.format(i))
                #assert(len(pts) == len(new_path))
                for l in lines[last_line:rng[0]]:
                    print l.strip()
                    last_line = rng[-1] + 1
                for np in new_path:
                    print str(np).strip()

            for l in lines[last_line:]:
                print l.strip()
                # p.sort(cmp=pt_dist_from(pts[0]))
                # print pts[0], p
        elif affine_:
            aff = Affine_Fit([(float(sys.argv[z]), float(sys.argv[z + 1])) for z in range(-12, -6, 2)],
                             [(float(sys.argv[z]), float(sys.argv[z + 1])) for z in range(-6, 0, 2)])
            print>>sys.stderr, "Transform is:\n", aff.To_Str()
            for lnum, line in enumerate(lines):
                if lnum not in ranges:
                    print line.strip()
                    continue

                line = find_transform_and_replace(line, aff)
                print line.strip()
        elif zlevel_:
            try:
                fdata = sys.argv[sys.argv.index("--zlevel") + 1]
            except ValueError as e:
                print>>sys.stderr, "after --zlevel must be a valid filename"
                exit(1)
            try:
                intpstep = float(sys.argv[sys.argv.index("--zlevel") + 2])
            except IndexError as e:
                print>>sys.stderr, "after --zlevel filename must be a valid interpolation step in milimeters"
                exit(1)
            except ValueError as e:
                print>>sys.stderr, "after --zlevel filename must be a valid interpolation step in milimeters"
                exit(1)
            with open(fdata) as f_data:
                # load data and turn them into floats
                #level_data = map(lambda __x: map(lambda __y: map(float, __y.split()), __x.split(',')), f_data.readlines())

                points = map(lambda x: map(float, x.split()),
                             f_data.readlines())

                points = sorted(points, key=lambda p: p[1])  # sort by y

                level_data = []
                for k, g in itertools.groupby(points, lambda x: x[1]):
                    level_data.append(list(g))

                # print(level_data)
                # print(len(level_data))
            z_level = 0.0  # in milimeters
            milling = False
            old_pt = None
            for lnum, line in enumerate(lines):
                if lnum not in ranges:
                    print line.strip()
                    continue
                #print>>sys.stderr, "z:", z_level

                if "G00" == command_name(line):
                    if "retract" in line:
                        line = find_and_replace(line, 'Z', retract_mm)
                        old_pt = None
                    else:
                        #line = find_zlevel_and_replace(
                        #    line, level_data, base_z_level=z_level).strip()
                        block_start_pt = Pt(line)

                elif "G01" == command_name(line):
                    p = Pt(line)
                    if p.z is not None:
                        z_level = p.z
                    if p.x is None and p.y is None:
                        if z_level < 0.0:
                            fake_line = 'G01 X{} Y{} Z{} F100 (begin)'.format(block_start_pt.x,
                                                             block_start_pt.y, z_level)

                            line = find_zlevel_and_replace(
                                fake_line, level_data, base_z_level=z_level).strip()
                            old_pt = block_start_pt
                            old_pt.z = p.z
#                            block_start_pt.z = 0.05
                            #print>>sys.stderr, l
                            #lpt = Pt(l)
                            #line = find_and_replace(line, 'Z', z_level)
                            #print>>sys.stderr, z_level, repr(line), repr(old_pt)
                        # we've have found MILLING END OR BEGIN
                    elif z_level < 0.0:
                        #print>>sys.stderr, "FUHA", z_level
                        if old_pt is not None:
                            # CUT IT!!!
                            STEP = intpstep  # in mm
                            segment_len = get_path_len([old_pt, p])[1]
                            #print>>sys.stderr, segment_len
                            if not segment_len == 0:
                                t_step = STEP / segment_len
                                for itera in xrange(1, int(segment_len / STEP)):
                                    intx_ = lerp(old_pt.x, p.x, itera * t_step)
                                    inty_ = lerp(old_pt.y, p.y, itera * t_step)
                                    #print>>sys.stderr, (intx_, inty_), (old_pt.x, p.x)
                                    newline = find_and_replace(line, 'X', intx_)
                                    newline = find_and_replace(newline, 'Y', inty_)
                                    newline = find_and_replace(newline, 'Z', z_level)
                                    newline = find_zlevel_and_replace(newline, level_data,
                                                                      base_z_level=z_level).strip()

                                    print('{} ( {} L:{} )'.format(newline, itera, lnum))
                                    # print "(chuj", itera, get_path_len([old_pt,
                                    # Pt(newline)])[1], ")"

                            line = find_zlevel_and_replace(
                                line, level_data, base_z_level=z_level)
                            old_pt = p
                        else:
                            line = find_zlevel_and_replace(
                                line, level_data, base_z_level=z_level)
                            old_pt = p

                print line.strip(), ' ( L:{} )'.format(lnum)
        else:
            scale = float(sys.argv[3])
            for lnum, line in enumerate(lines):
                if lnum not in ranges:
                    print line.strip()
                    continue

                line = find_scale_and_replace(line, "X", scale)
                line = find_scale_and_replace(line, "Y", scale)
                print line.strip()
