import sklearn_crfsuite
import xml
import sklearn
from sklearn.utils import shuffle
import re
import os
import argparse
import pickle
import xml.etree.ElementTree as ET
from sklearn.metrics import make_scorer
from sklearn_crfsuite import metrics
import scipy.stats
from sklearn.model_selection import RandomizedSearchCV


def remove_delimiters(input_text: str, delim: list) -> str:
    """
    remove all delimiters from the input string
    """
    list_input = [char for char in input_text if char not in delim]
    return ''.join(list_input)


def build_tag_seq(input_words: list, delimiter_mapper: dict) -> list:
    """
    This function generates a sequence of labels for the input text data
    delimiter_mapper specifies the delimiters used and the labels for 
    the characters immediately after the delimiter
    """
    internal_tag = "IN"
    beginning_tag = "BEG"
    output_tags = []
    word_i = 0
    while word_i < len(input_words):
        curr_tag_seq = []
        word = input_words[word_i]
        character_i = 0
        while character_i < len(word):
            character = word[character_i]
            if character in delimiter_mapper.keys():
                curr_tag_seq.append(delimiter_mapper[character])
                character_i += 1
            elif character_i == 0:
                curr_tag_seq.append(beginning_tag)
            else:
                curr_tag_seq.append(internal_tag)
                
            character_i += 1
                
        output_tags.append(curr_tag_seq)
        word_i += 1
    
    return output_tags


def build_dataset(xml_file: str, delimiter_mapper: dict):
    """
    Build a dataset from an xml file, the dataset consists of a sequence of tags 
    and a sequence of words with all delimiters removed
    """
    tree = ET.parse(xml_file)
    doc_root = tree.getroot()
    segmentations = doc_root.findall('.//seg')
    segmentations = [seg.text.split() for seg in segmentations if seg.text != None]
    tags = [build_tag_seq(word, delimiter_mapper) for word in segmentations]
    scrubbed_text = []
    for sent in segmentations:
        scrubbed_text.append([remove_delimiters(word, delimiter_mapper.keys()) for word in sent])

    return scrubbed_text, tags, segmentations

def char_2_feats(word: str, i: int)->dict:
    features = {
            'lower':word[i].lower(),
            'last_char':word[-1],
    }
    if len(word) > 1:
        features['2last_char'] = word[-2]

    if i > 0:
        features['prev'] = word[i-1]
        
    else:
        features['prev'] = 'BOW'
    
    if i < len(word) - 1:
        features['next'] = word[i+1]
        
    else:
        features['next'] = 'EOW'
        
    return features

def word_2_feats(word):
    return [char_2_feats(word, i) for i in range(len(word))]
    
def verify_data(features, tags, delim_word):
    if len(features) != len(tags):
        print("The number of words in the features and ")
        print("the number of words in the tags do not match")
    
    else:
        for i in range(len(features)):
            if len(features[i]) != len(tags[i]):
                print(str(len(features[i])) + " :::::::::: " + str(len(tags[i])))
                print(features[i])
                print(tags[i])
                print(delim_word[i])
                print()

                
def main():
    out_pkl = "dataset.pkl"
    delimiter_mapper = {"-":"MOR",
                        "=":"CLI",
                        "+":"CON"}
                        #"~":"MOR"} # unsure about this last one
    words = []
    tags = []
    delim_words = []
    paths = os.walk("ATMO/Comments_Removed/")
    for root, directory, filespecs in paths:
        for filespec in filespecs:
            print("Processing: " + filespec)
            path = os.path.join(root, filespec)
            single_text, single_tags, single_seg = build_dataset(path, delimiter_mapper)
            print("Adding " + str(len(single_text)) + " lines")
            words += single_text
            print("Total: " + str(len(words)) + " lines")
            tags += single_tags
            delim_words += single_seg
    
    flat_tags = [tag for sent in tags for tag in sent]
    flat_words = [word for sent in words for word in sent]
    flat_delim = [word for sent in delim_words for word in sent]
    print(flat_tags[0:5])
    print(flat_words[0:5])
    verify_data(flat_words, flat_tags, delim_word=flat_delim)
    features = [word_2_feats(word) for word in flat_words]
    crf = sklearn_crfsuite.CRF(
                    algorithm='lbfgs',
                    c1=0.1,
                    c2=0.1,
                    max_iterations=100,
                    all_possible_transitions=True
            )
    print(flat_words[0])
    print(flat_tags[0])
    #verify_data(features, flat_tags)
    features, flat_tags = shuffle(features, flat_tags)
    train_features = features[0:12000]
    train_tags = flat_tags[0:12000]
    test_features = features[12000:]
    test_tags = flat_tags[12000:]
    f1_scorer = make_scorer(metrics.flat_f1_score,
                        average='weighted')
    params_space = {'c1':scipy.stats.expon(scale=0.5),
                    'c2':scipy.stats.expon(scale=0.05)}
    rs = RandomizedSearchCV(crf, params_space, cv=5, verbose=8, n_jobs=8, n_iter=100, scoring=f1_scorer)
    rs.fit(train_features, train_tags)
    pred_tags = rs.predict(test_features)
    print(metrics.flat_classification_report(test_tags, pred_tags, digits=8))

if __name__ == '__main__':
    main()
