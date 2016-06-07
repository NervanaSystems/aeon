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
        print('opt {0}, arg {1}').format(opt,arg)
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
