import argparse
from sklearn.metrics import confusion_matrix
from sklearn_crfsuite.metrics import flat_classification_report
import json

parser = argparse.ArgumentParser(description="Process the prediction output from allennlp and generate a confusion matrix and classification report")
parser.add_argument("--predictions-file", dest="in_file", type=str,
                    help="The path to the predictions file from allennlp")
parser.add_argument("--gold-file", dest="gold_file", type=str,
                    help="The path to the gold standard file")
args = parser.parse_args()
labels = []
predictions = []
with open(args.in_file) as in_file:
    for line in in_file:
        res_tags = json.loads(line)["tags"]
        predictions.append(res_tags)

with open(args.gold_file) as gold_file:
    for line in gold_file:
        line_labels = []
        split_line = line.split('\t')
        for word_chunk in split_line:
            try:
                tag = word_chunk.split('###')[1]
                line_labels.append(tag)
            except IndexError:
                pass

        labels.append(line_labels)
            
print(flat_classification_report(labels, predictions, digits=8))
