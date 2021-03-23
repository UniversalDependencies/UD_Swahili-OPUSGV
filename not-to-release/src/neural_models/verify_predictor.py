import argparse
from tqdm import tqdm
import allennlp
from allennlp.predictors import Predictor

from src_new import *

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="Lemmatizer model trained from the Helsinki corpus")
args = parser.parse_args()


def load_predictor(filepath: str) -> MultilabelSentenceTaggerPredictor:
    predictor = Predictor.from_path(filepath, 
                                    predictor_name="multilabel_sentence_tagger")
    return predictor

    
def run_model(input_str:str, predictor: MultilabelSentenceTaggerPredictor):
    output = predictor.predict(input_str)
    return output['tags']


input_text = "Ninapenda nyanya kila siku njema"
predictor = load_predictor(args.model)
output = run_model(input_text, predictor)
print(output)
