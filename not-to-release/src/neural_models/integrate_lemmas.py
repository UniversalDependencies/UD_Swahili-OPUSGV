import argparse
from tqdm import tqdm
import allennlp
from allennlp.predictors import Predictor
from allennlp.predictors import Seq2SeqPredictor

parser = argparse.ArgumentParser()
parser.add_argument("--input-file", "-i")
parser.add_argument("--output-file", "-o")
parser.add_argument("--model", help="Lemmatizer model trained from the Helsinki corpus")
args = parser.parse_args()

def load_predictor(filepath: str) -> Seq2SeqPredictor:
    predictor = Predictor.from_path(filepath, 
                                    predictor_name="seq2seq")
    return predictor
    
def run_model(input_str:str, predictor: Seq2SeqPredictor):
    output = predictor.predict(input_str)
    characters = output['predicted_tokens']
    return "".join(characters)

predictor = load_predictor(args.model)
with open(args.input_file) as input_file:
    with open(args.output_file, "w") as output_file:
        for line in tqdm(input_file):
            line = line.strip()
            # Skip comment lines
            if line.startswith("#"):
                output_file.write(line + "\n")
            # is line blank (indicting a new sentence)
            elif not line:
                output_file.write("\n")
            else:
                split_line = line.split("\t")
                word = split_line[1]
                pos = split_line[3]
                lemma = ""
                if pos == "PUNCT":
                    lemma = word
                else:
                    model_input = word + " " + pos
                    lemma = run_model(model_input, predictor)
                # reassemble line
                split_line[2] = lemma
                line = "\t".join(split_line)
                output_file.write(line + "\n")
