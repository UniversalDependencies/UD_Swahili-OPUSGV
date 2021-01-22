"""
This script generates the file that is required for prediction by
allennlp's predict command. 

In essence, this reads in every line in the input and creates a list where each entry in the list is a dictionary that has one key `"sentence"` with one value, which is the sentence for that line in the input file.
"""
import json
import argparse

parser = argparse.ArgumentParser(description="Generate json file for prediction from input file with oen tweet per line")
parser.add_argument("--in_file", dest="in_file", type=str, 
                    help="The path to the file to process")
parser.add_argument("--out_file", dest="out_file", type=str,
                    help="The path to write the output json file to")

args = parser.parse_args()

sentences = []
with open(args.in_file) as in_file:
    for line in in_file:
        sent_dict = {"sentence": line}
        sentences.append(sent_dict)

print("Writing json file " + args.out_file + "...")                         
with open(args.out_file, "w") as out_file:
    json.dump(sentences, out_file)
