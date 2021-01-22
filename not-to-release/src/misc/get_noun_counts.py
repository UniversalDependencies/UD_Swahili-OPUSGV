import argparse
from sklearn.metrics import confusion_matrix
from sklearn_crfsuite.metrics import flat_classification_report
import json
from collections import Counter

parser = argparse.ArgumentParser(description="Process the prediction output from allennlp and the gold files from the helsinki corpus and generate a count of the number of nouns")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--predictions-file", dest="in_file", type=str,
                    help="The path to the predictions file from allennlp")
group.add_argument("--gold-file", dest="gold_file", type=str,
                    help="The path to the gold standard file")
args = parser.parse_args()
nouns = []
if args.in_file:
    with open(args.in_file) as in_file:
        for line in in_file:
            res_tags = json.loads(line)["tags"]
            res_words = json.loads(line)["words"]
            for tag_i in range(len(res_tags)):
                if res_tags[tag_i] == "NOUN":
                    nouns.append(res_words[tag_i])

if args.gold_file:
    with open(args.gold_file) as gold_file:
        for line in gold_file:
            line_labels = []
            split_line = line.split('\t')
            for word_chunk in split_line:
                if "###" in word_chunk:
                    word, tag = word_chunk.split('###')
                if tag == "NOUN":
                    nouns.append(word)
           
res = Counter(nouns)
with open("noun_counts.txt", "w") as out_file:
    for noun in sorted(res.items(), key=lambda x: x[1]):
        noun_txt = noun[0]
        noun_count = noun[1]
        out_file.write(noun_txt + "\t" + str(noun_count) + "\n")
