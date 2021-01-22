from typing import List, Dict
from copy import deepcopy

from overrides import overrides
import numpy

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.predictors.sentence_tagger import SentenceTaggerPredictor
import warnings

def remove_logits_keys(model_dict):
    """
    Simple function to remove the logits and class probabilities
    keys if they are present. This reduces the file size of the predictions
    data drastically.
    """
    if "logits" in model_dict.keys():
        del model_dict["logits"]

    if "class_probabilities" in model_dict.keys():
        del model_dict["class_probabilities"]

def remove_logits(model_output):
    if type(model_output) == dict:
        remove_logits_keys(model_output)

    elif type(model_output) == list:
        for return_val in model_output:
            if type(return_val) == dict:
                remove_logits_keys(return_val)


@Predictor.register("sentence-tagger-gen")
class GeneralSentenceTaggerPredictor(Predictor):
    """
    Predictor for any model that takes in a sentence and returns
    a single set of tags for it.  In particular, it can be used with
    the :class:`~allennlp.models.crf_tagger.CrfTagger` model
    and also
    the :class:`~allennlp.models.simple_tagger.SimpleTagger` model.
    """

    def __init__(
        self, 
        model: Model, 
        dataset_reader: DatasetReader, 
        tokenizer: Tokenizer = WordTokenizer(SpacyWordSplitter(language="en_core_web_sm", pos_tags=True))
    ) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = tokenizer
        warnings.warn("Creating general sentence tagger model")
    
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        remove_logits(outputs)
        return sanitize(outputs)

    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        remove_logits(outputs)
        return sanitize(outputs)

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        result = SentenceTaggerPredictor.predict_json(self, inputs)
        remove_logits(result)
        return result

    def predict_batch_json(self, inputs: JsonDict) -> JsonDict:
        result = SentenceTaggerPredictor.predict_batch_json(self, inputs)
        for ret in result:
            remove_logits(ret)
        return result

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"words"`` to the output.
        """
        sentence = json_dict["sentence"]
        tokens = self._tokenizer.tokenize(sentence)
        return self._dataset_reader.text_to_instance(tokens)

    @overrides
    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ) -> List[Instance]:
        """
        This function handles any kind of sequence of tags. 
        
        However, no special considerations with regard to BIOU tags are taken.
        """
        predicted_tags = outputs["tags"]
        predicted_spans = []

        i = 0
        while i < len(predicted_tags):
            tag = predicted_tags[i]
            # if its a U, add it to the list
            if tag[0] == "U":
                current_tags = [t if idx == i else "O" for idx, t in enumerate(predicted_tags)]
                predicted_spans.append(current_tags)
            # if its a B, keep going until you hit an L.
            elif tag[0] == "B":
                begin_idx = i
                while tag[0] != "L":
                    i += 1
                    tag = predicted_tags[i]
                end_idx = i
                current_tags = [
                    t if begin_idx <= idx <= end_idx else "O"
                    for idx, t in enumerate(predicted_tags)
                ]
                predicted_spans.append(current_tags)
            i += 1

        # Creates a new instance for each contiguous tag
        instances = []
        for labels in predicted_spans:
            new_instance = deepcopy(instance)
            text_field: TextField = instance["tokens"]  # type: ignore
            new_instance.add_field(
                "tags", SequenceLabelField(labels, text_field), self._model.vocab
            )
            instances.append(new_instance)
        instances.reverse()  # NER tags are in the opposite order as desired for the interpret UI

        return instances
