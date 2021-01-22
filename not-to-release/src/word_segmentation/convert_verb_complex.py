import argparse
from sklearn.metrics import confusion_matrix
from sklearn_crfsuite.metrics import flat_classification_report
import json
from collections import Counter
import sys
sys.path.append("..")
from corpus_tools import corpus_tools
import re

def split_verb(verb:str):
    """
    Function that uses regexes to determine the place to split the subject marker + 
    TAM marker from the object marker and verb root.
    
    Params:
    ========
    `verb` :    `String`
                The verb string to process
    """

    subject_prefixes = ["a","u","ni","tu","m", "wa"]
    # ha has to be at the end because otherwise we'll get early matches since regexes check options in the order listed
    negated_subject_prefixes = ["si","hu","hatu","ham","hawa", "ha"]
    noun_class_subject_prefixes = ["a", "wa", "u", "i","li", "ya", 
                                   "ki","vi", "i", "zi","u", "ku","pa","ku","m"]
    negated_noun_class_subject_prefixes = ["hau","hai","hali","haya","haki",
                                           "havi","hai","hazi","hau","haku",
                                           "hapa", "haku","ham"]
    noun_class_object_prefixes = ["m", "wa", "m", "i","li","ya","ki",
                                  "vi","i","zi","u","zi","u","zi","u",
                                  "ku","pa","ku","mu"]
    tense_markers = ["na", "ta", "li", "ngeli","ki"]
    all_subject_prefixes =  subject_prefixes + noun_class_subject_prefixes #+ negated_noun_class_subject_prefixes + negated_subject_prefixes
    neg_subject_prefixes =  negated_noun_class_subject_prefixes + negated_subject_prefixes
    # if the input verb begins with a negated noun class marker then tense is optional, otherwise it is required.
    # if negation is used and we ahve ki (which is ambiguous and can be either a tense marker or object marker) we should prefer
    # the object marker version since negated ki as a tense marker is very infrequent
    # I'm doing this in a series of steps because a single regex is hard to debug and fix in the future.
    subject_prefix_regex = "(" + "|".join(all_subject_prefixes) + ")"
    neg_subject_prefix_regex = "(" + "|".join(neg_subject_prefixes) + ")"
    tam_regex = "(" + "|".join(tense_markers) + ")"
    object_regex = "(" + "|".join(noun_class_object_prefixes) + ")"
    # check if we have negation
    if re.match("^" + neg_subject_prefix_regex + ".*", verb):
        print("We have negation")
        # check if we have neg + ki
        if re.match("^" + neg_subject_prefix_regex + "ki", verb):
            print("We have ki")
            verb_w_whitespace = re.sub("^" + neg_subject_prefix_regex, r"\1 ", verb)
        else:
            print("we do not have ki")
            regex = "^" + neg_subject_prefix_regex + tam_regex + "?" + object_regex + "?"
            print(regex)
            verb_w_whitespace = re.sub(regex, r"\1\2 \3", verb)
    else:
        regex = "^" + subject_prefix_regex + tam_regex + object_regex + "?"
        print(regex)
        verb_w_whitespace = re.sub(regex, r"\1\2 \3", verb)

    split_verb = verb_w_whitespace.split()
    print(split_verb)
    assert len(split_verb) == 2
    return split_verb

def parse_morph_features(morph_tags:str):
    """
    Parses the morphological features from a UD morph features string into a dictionary
    of attributes and values.
    
    Params:
    =======
    `morph_tags`:   `String`
                    The morph string to process. These are of the format Attribute1=Value1|Attribute2=Value2 etc.
                    
    Returns:
    =======
    `parsed_morph_tags` :   `Dict`
                            The parsed morph string in dictionary format.
    """
    split_tags = morph_tags.split("|")
    tag_dict = {}
    for tag in split_tags:
        attribute, value = tag.split("=")
        tag_dict[attribute] = value
        
    return tag_dict
    
def separate_morph_tags(morph_tags:str):
    """
    Function that separates the morph tags out and detemines which tags belong to the
    split off AUX and which belong to the verb stem.
    
    Params:
    =======
    `morph_tags`:   `String`
                    The morph tag string to process. As with all UD morph tags this should
                    consist of a number of attribute value pairs separated by '|'
    """
    pass
