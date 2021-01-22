[this](https://github.com/allenai/allennlp/blob/dcd1d25e3f3dd0f672de32403080d75bcfe005b9/allennlp/tests/fixtures/encoder_decoder/simple_seq2seq/experiment.json) is an example of simple seq2seq model using the seq2seq reader

```
allennlp predict --include-package src --predictor sentence-tagger trained_models/swh_func_model.tar.gz ../data/global_voices/en-sw/GlobalVoices.en-sw.sw.json --cuda-device 1 --output-file ../data/allennlp-global-voices/func_tag.json_lines --silent
```

## Obtaining predictions for helsinki data

The key is `--use-dataset-reader`
```
allennlp predict --include-package src --predictor sentence-tagger-gen trained_models/swh_ud_pos.tgz ../data/allennlp-helsinki-ud-pos/test.txt --use-dataset-reader --cuda-device 1 --output-file ../data/allennlp-helsinki-ud-pos/predict.json_lines --silent
```
