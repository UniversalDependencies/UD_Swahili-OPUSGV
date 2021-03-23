from typing import Dict, Optional, List, Any

import numpy
from overrides import overrides
import torch
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from src_new.util import sequence_multilabel_loss_with_logits
from allennlp.training.metrics import FBetaMultiLabelMeasure, SpanBasedF1Measure


@Model.register("multilabel_tagger")
class MultiLabelTagger(Model):
    """
    This `MultiLabelTagger` simply encodes a sequence of text with a stacked `Seq2SeqEncoder`, then
    predicts multiple labels for each token in the sequence.

    Registered as a `Model` with name "multilabel_tagger".

    # Parameters

    vocab : `Vocabulary`, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : `TextFieldEmbedder`, required
        Used to embed the `tokens` `TextField` we get as input to the model.
    encoder : `Seq2SeqEncoder`
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and predicting output tags.
    calculate_span_f1 : `bool`, optional (default=`None`)
        Calculate span-level F1 metrics during training. If this is `True`, then
        `label_encoding` is required. If `None` and
        label_encoding is specified, this is set to `True`.
        If `None` and label_encoding is not specified, it defaults
        to `False`.
    label_encoding : `str`, optional (default=`None`)
        Label encoding to use when calculating span f1.
        Valid options are "BIO", "BIOUL", "IOB1", "BMES".
        Required if `calculate_span_f1` is true.
    label_namespace : `str`, optional (default=`labels`)
        This is needed to compute the SpanBasedF1Measure metric, if desired.
        Unless you did something unusual, the default value should be what you want.
    verbose_metrics : `bool`, optional (default = `False`)
        If true, metrics will be returned per label class in addition
        to the overall statistics.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        Used to initialize the model parameters.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        calculate_span_f1: bool = None,
        label_encoding: Optional[str] = None,
        label_namespace: str = "labels",
        verbose_metrics: bool = False,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)

        self.label_namespace = label_namespace
        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size(label_namespace)
        self.encoder = encoder
        self._verbose_metrics = verbose_metrics
        self.tag_projection_layer = TimeDistributed(
            Linear(self.encoder.get_output_dim(), self.num_classes)
        )

        check_dimensions_match(
            text_field_embedder.get_output_dim(),
            encoder.get_input_dim(),
            "text field embedding dim",
            "encoder input dim",
        )

        self.metrics = {
            "fbeta" : FBetaMultiLabelMeasure(average="macro")
        }

        # We keep calculate_span_f1 as a constructor argument for API consistency with
        # the CrfTagger, even it is redundant in this class
        # (label_encoding serves the same purpose).
        if calculate_span_f1 is None:
            calculate_span_f1 = label_encoding is not None

        self.calculate_span_f1 = calculate_span_f1
        self._f1_metric: Optional[SpanBasedF1Measure] = None
        if calculate_span_f1:
            if not label_encoding:
                raise ConfigurationError(
                    "calculate_span_f1 is True, but no label_encoding was specified."
                )
            self._f1_metric = SpanBasedF1Measure(
                vocab, tag_namespace=label_namespace, label_encoding=label_encoding
            )

        initializer(self)

    @overrides
    def forward(
        self,  # type: ignore
        tokens: TextFieldTensors,
        tags: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
        ignore_loss_on_o_tags: bool = False,
    ) -> Dict[str, torch.Tensor]:

        """
        # Parameters

        tokens : `TextFieldTensors`, required
            The output of `TextField.as_array()`, which should typically be passed directly to a
            `TextFieldEmbedder`. This output is a dictionary mapping keys to `TokenIndexer`
            tensors.  At its most basic, using a `SingleIdTokenIndexer` this is : `{"tokens":
            Tensor(batch_size, num_tokens)}`. This dictionary will have the same keys as were used
            for the `TokenIndexers` when you created the `TextField` representing your
            sequence.  The dictionary is designed to be passed directly to a `TextFieldEmbedder`,
            which knows how to combine different word representations into a single vector per
            token in your input.
        tags : `torch.LongTensor`, optional (default = `None`)
            A torch tensor representing the sequence of integer gold class labels of shape
            `(batch_size, num_tokens)`.
        metadata : `List[Dict[str, Any]]`, optional, (default = `None`)
            metadata containing the original words in the sentence to be tagged under a 'words' key.
        ignore_loss_on_o_tags : `bool`, optional (default = `False`)
            If True, we compute the loss only for actual spans in `tags`, and not on `O` tokens.
            This is useful for computing gradients of the loss on a _single span_, for
            interpretation / attacking.

        # Returns

        An output dictionary consisting of:
            - `logits` (`torch.FloatTensor`) :
                A tensor of shape `(batch_size, num_tokens, tag_vocab_size)` representing
                unnormalised log probabilities of the tag classes.
            - `label_probabilities` (`torch.FloatTensor`) :
                A tensor of shape `(batch_size, num_tokens, tag_vocab_size)` representing
                a distribution of the tag classes per word. Tags are positive if they exceed a threshold
                (typically 0.5).
            - `loss` (`torch.FloatTensor`, optional) :
                A scalar loss to be optimised.

        """
        embedded_text_input = self.text_field_embedder(tokens)
        batch_size, sequence_length, _ = embedded_text_input.size()
        mask = get_text_field_mask(tokens)
        encoded_text = self.encoder(embedded_text_input, mask)

        logits = self.tag_projection_layer(encoded_text)
        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = torch.sigmoid(reshaped_log_probs).view(
            [batch_size, sequence_length, self.num_classes]
        )
        output_dict = {"logits": logits, "label_probabilities": class_probabilities}

        if tags is not None:
            if ignore_loss_on_o_tags:
                o_tag_index = self.vocab.get_token_index("O", namespace=self.label_namespace)
                tag_mask = mask & (tags != o_tag_index)
            else:
                tag_mask = mask
            loss = sequence_multilabel_loss_with_logits(logits, tags, tag_mask)
            for metric in self.metrics.values():
                logits_flat = logits.view(-1, logits.size(-1))
                tags_flat = tags.view(-1, tags.size(-1))
                mask = mask.unsqueeze(-1).repeat(1, 1, logits_flat.size(-1))
                mask = mask.view(-1, mask.size(-1))
                metric(logits_flat, tags_flat, mask)
            if self.calculate_span_f1:
                self._f1_metric(logits, tags, mask)  # type: ignore
            output_dict["loss"] = loss

        if metadata is not None:
            output_dict["words"] = [x["words"] for x in metadata]
        return output_dict

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Does a simple position-wise argmax over each token, converts indices to string labels, and
        adds a `"tags"` key to the dictionary with the result.
        """
        all_predictions = output_dict["label_probabilities"]
        all_predictions = all_predictions.cpu().data.numpy()
        # If we have a batch dimension, split off each batch into a separate ndarray in a list of ndarrays.
        if all_predictions.ndim == 3:
            predictions_list = [all_predictions[i] for i in range(all_predictions.shape[0])]
        else:
            predictions_list = [all_predictions]
        all_tags = []
        for predictions in predictions_list:
            # Argmax_indices is an ndarray where the first index corresponds to
            # the token id in the input and the second is the index of the predicted label.
            argmax_indices = numpy.argwhere(predictions>0.5)
            words = [[] for word in numpy.unique(argmax_indices[:, 0])]
            for word_index, tag_index in argmax_indices:
                single_string_token = self.vocab.get_token_from_index(tag_index, namespace=self.label_namespace)
                words[word_index].append(single_string_token)
            
            all_tags.append(words)
        output_dict["tags"] = all_tags
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {}
        for metric_name, metric in self.metrics.items():
            metrics_to_return.update(metric.get_metric(reset))

        if self.calculate_span_f1:
            f1_dict = self._f1_metric.get_metric(reset)  # type: ignore
            if self._verbose_metrics:
                metrics_to_return.update(f1_dict)
            else:
                metrics_to_return.update({x: y for x, y in f1_dict.items() if "overall" in x})
        return metrics_to_return

    default_predictor = "sentence_tagger"
