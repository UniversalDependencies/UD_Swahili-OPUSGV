import stanfordnlp

def generate_conll(sentences, out_file_path:str):
    """
    This generates a conll file given a list of word
    """
    with open(out_file_path, "w") as out_file:
        for sentence in sentences:
            for word in sentence.words:
                if word.lemma == None:
                    word.lemma = "_"

                line = str(word.index) + "\t"
                line += word.text + "\t" 
                line += word.lemma + "\t" 
                line += word.upos + "\t" 
                line += word.xpos + "\t" 
                line += word.feats + "\t" 
                line += str(word.governor) + "\t" 
                line += str(word.dependency_relation) + "\n"
                out_file.write(line)
            out_file.write("\n")
            

def main():
    #stanfordnlp.download('en')
    nlp = stanfordnlp.Pipeline(tokenize_batch_size=256, mwt_batch_size=64, pos_batch_size=256, lemma_batch_size=256, depparse_batch_size=256)
    text = ""
    with open("global_voices/en-sw/GlobalVoices.en-sw.en") as in_txt:
        lines = in_txt.readlines()
        for line in lines:
            text += line + "\n"
        
    doc = nlp(text)
    generate_conll(doc.sentences, "global_voices/en-sw/GlobalVoices.en-sw.en.conll")
    
main()
