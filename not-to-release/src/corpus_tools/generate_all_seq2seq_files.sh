files=`ls data/hcs-na-v2/`
for file in $files
do
	python3 generate_seq2seq_file.py --in_json data/hcs-na-v2/$file --out_file data/allennlp-helsinki-lemma/$file > $file-unlabeled-log.txt
done
