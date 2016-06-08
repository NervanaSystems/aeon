#!/usr/bin/python

import json
import glob
import sys
import getopt
import collections
import os
from os.path import isfile, join
import xml.etree.ElementTree as et
from collections import defaultdict

# http://stackoverflow.com/questions/7684333/converting-xml-to-dictionary-using-elementtree
def etree_to_dict(t):
    d = {t.tag: {} if t.attrib else None}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(etree_to_dict, children):
            for k, v in dc.iteritems():
                dd[k].append(v)
        d = {t.tag: {k:v[0] if len(v) == 1 else v for k, v in dd.iteritems()}}
    if t.attrib:
        d[t.tag].update(('@' + k, v) for k, v in t.attrib.iteritems())
    if t.text:
        text = t.text.strip()
        if children or t.attrib:
            if text:
              d[t.tag]['#text'] = text
        else:
            d[t.tag] = text
    return d

def validate_metadata(jobj,file):
    boxlist = jobj['object']
    if not isinstance(boxlist,collections.Sequence):
        print('{0} is not a sequence').format(file)
        return False
    # print("{0} has {1} boxes").format(jobj['filename'],len(boxlist))
    index = 0;
    for box in boxlist:
        if 'part' in box:
            parts = box['part']
            if not isinstance(parts,collections.Sequence):
                print('parts {0} is not a sequence').format(file)
                return False
        index += 1
    return True

def convert_pascal_to_json(input_path,output_path):
    #onlyfiles = [f for f in listdir(input_path) if isfile(join(input_path, f)) && file.endswith('.xml')]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    onlyfiles = glob.glob(join(input_path,'*.xml'))
    onlyfiles.sort()
    for file in onlyfiles:
        outfile = join(output_path,os.path.basename(file))
        outfile = os.path.splitext(outfile)[0]+'.json'
        print(outfile)
        trimmed = parse_single_file(join(input_path,file))
        if validate_metadata(trimmed,file):
            result = json.dumps(trimmed, sort_keys=True, indent=4, separators=(',', ': '))
            f = open(outfile,'w')
            f.write(result)
        else:
            print('error parsing metadata {0}').format(file)
        #print(result)

def parse_single_file(path):
    tree = et.parse(path)
    root = tree.getroot()
    d = etree_to_dict(root)
    trimmed = d['annotation']
    olist = trimmed['object']
    if not isinstance(olist,collections.Sequence):
        trimmed['object'] = [olist];
        olist = trimmed['object']
    size = trimmed['size']

    # Add version number to json
    trimmed['version'] = {'major': 1, 'minor': 0}

    # convert all numbers from string representation to number so json does not quote them
    # all of the bounding box numbers are one based so subtract 1
    size['width'] = int(size['width'])
    size['height'] = int(size['height'])
    size['depth'] = int(size['depth'])
    width = trimmed['size']['width']
    height = trimmed['size']['height']
    for obj in olist:
        obj['difficult'] = int(obj['difficult']) != 0
        obj['truncated'] = int(obj['truncated']) != 0
        box = obj['bndbox']
        box['xmax'] = int(box['xmax'])-1
        box['xmin'] = int(box['xmin'])-1
        box['ymax'] = int(box['ymax'])-1
        box['ymin'] = int(box['ymin'])-1
        if 'part' in obj:
            for part in obj['part']:
                box = part['bndbox']
                box['xmax'] = float(box['xmax'])-1
                box['xmin'] = float(box['xmin'])-1
                box['ymax'] = float(box['ymax'])-1
                box['ymin'] = float(box['ymin'])-1
        xmax = box['xmax']
        xmin = box['xmin']
        ymax = box['ymax']
        ymin = box['ymin']
        if xmax > width-1:
            print('xmax {0} exceeds width {1}').format(xmax,width)
        if xmin < 0:
            print('xmin {0} exceeds width {1}').format(xmin,width)
        if ymax > height-1:
            print('ymax {0} exceeds width {1}').format(ymax,height)
        if ymin < 0:
            print('ymin {0} exceeds width {1}').format(ymin,height)
    return trimmed

def main(argv):
    input_path = ''
    output_path = ''
    parse_file = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:p:")
    except getopt.GetoptError:
        print 'ingest.py -i <input> -o <output>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'ingest.py -i <input> -o <output>'
            sys.exit()
        elif opt in ("-i", "--input"):
            input_path = arg
        elif opt in ("-o", "--output"):
            output_path = arg
        elif opt in ("-p", "--parse"):
            parse_file = arg

    print(parse_file)
    if parse_file:
        parsed = parse_single_file(parse_file)
        json1 = json.dumps(parsed, sort_keys=True, indent=4, separators=(',', ': '))
        print(json1)
    elif input_path:
        convert_pascal_to_json(input_path,output_path)

if __name__ == "__main__":
   main(sys.argv[1:])

# file = '/usr/local/data/VOCdevkit/VOC2007/Annotations/006637.xml'
# tree = et.parse(file)

# root = tree.getroot()
# d = etree_to_dict(root)

# # et.dump(tree)
# json2 = d['annotation']
# json1 = json.dumps(json2, sort_keys=True, indent=4, separators=(',', ': '))
# print(json1)

# path = '/usr/local/data/VOCdevkit/VOC2007/Annotations/*.xml'
# convert_pascal_to_json(path)
