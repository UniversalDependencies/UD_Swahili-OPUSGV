__author__ = "Kenneth Steimel"
__email__ = "ksteimel@iu.edu"
__date__ = "September-2019"


import xml.etree.ElementTree as ET
import sys
import argparse
import re
import os
import swahili_funcs

parser = argparse.ArgumentParser(description="Generate sequence tagging files from the Helsinki Corpus's json format")
parser.add_argument("--in_json", dest="in_file", type=str,
                    help="the input file from the Helsinki corpus to process")
parser.add_argument("--out_file", dest="out_file", type=str,
                    help="the filepath where the sequence tagging format file should be written")
parser.add_argument("--label", dest="label", type=str, required=False,
                    help="the label to use when generating the sequence tag output. Valid options are 'pos' and 'func'")
args = parser.parse_args()

doc = swahili_funcs.Document()
doc.from_json(args.in_file)
doc.set_verb_tags()
doc.set_noun_classes()
doc.set_upos_tags()
print(len(doc.get_sents()))
doc.remove_missing()
print(len(doc.get_sents()))
doc.to_seq_tag_format(args.out_file, label=args.label)
