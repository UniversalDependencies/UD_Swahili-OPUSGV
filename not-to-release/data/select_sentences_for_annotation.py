import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Randomly select sentences from the entire corpus using length constraints")
parser.add_argument("--conll-file", dest="conll_file", type=str,
                    help="The path to the full conll file to read in")
parser.add_argument("--min-length", dest="min_length", type=int, default=0,
                    help="Sentences shorter than this length will be discarded")
parser.add_argument("--max-length", dest="max_length", type=int, default=20,
                    help="Sentences longer than this length will be discarded")
parser.add_argument("--n-select", dest="n_select", type=int, default=200,
                    help="The number of sentences to randomly select")

args = parser.parse_args()

corpus = []
sampled_corpus = []
with open(args.conll_file) as in_conll:
    sent = []
    for line in in_conll:
        if line == "\n":
            corpus.append(sent)
            sent = []
        else:
            sent.append(line)
   
corpus_indexes = [sent_i for sent_i in range(len(corpus)) if len(corpus[sent_i]) > args.min_length and len(corpus[sent_i]) < args.max_length]
np.random.shuffle(corpus_indexes)
sampled_corpus_indexes = corpus_indexes[0:args.n_select]
for sent_i in sampled_corpus_indexes:
    sent = corpus[sent_i]
    print('# ' + str(sent_i)) # including the index of the sampled sentence in the original file
    for word in sent:
        print(word, end="")
    
    print()
