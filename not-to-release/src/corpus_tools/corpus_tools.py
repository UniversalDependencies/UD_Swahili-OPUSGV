import re
import json
import warnings

def format_splat(format_string, *args):
        """
        This is a simple helper function to allow me to call format
        on an array. This is crucial for allowing interlinear_print to work
        correctly
        """
        return format_string.format(*args)

class SwahiliTag:
    """
    This is the main class that will contain the annotated swahili data
    """
    def __init__(self, word, pos, msd, cls=None, tam=None, subj=None,
                 obj=None, rel=None, gloss=None, func=None, lemma=None, msd_extra=None):
        self.__word = word
        self.__pos = pos
        self.__msd = msd
        self.__msd_extra = msd_extra
        self.__cls = cls
        self.__tam = tam
        self.__subj = subj
        self.__obj = obj
        self.__rel = rel
        #self.__gloss = gloss
        self.__func = func
        self.__lemma = lemma
        
    def get_morph_feats(self) -> str:
        """
        Generate a string representing the morphological information pulled from the msd tags
        """
        delim_str = "||"
        morph_str = (str(self.__cls) + 
                    delim_str +  
                    str(self.__tam) + 
                    delim_str + 
                    str(self.__subj) + 
                    delim_str + 
                    str(self.__obj) + 
                    delim_str + 
                    str(self.__rel)) 
        return morph_str 
    
    def get_noun_ud_morph_feats(self) -> list:
        """
        Get ud morphological features for nouns.
        
        per ud version 2 
        >The words should be annotated with the Number feature 
        >in addition to NounClass, despite the fact that people who know Bantu 
        >could infer the number from the noun class.
        
        """
        feats = []
        class_matches = re.findall(r"\s*([0-9/]+)-((?:PLSG)|(?:SG)|(?:PL))\s*", self.__msd)
        if "MASS" in self.__msd_extra:
            # this is reserved for mass nouns in class 11
            feats.append("Number=Coll")
        if class_matches:
            class_res = class_matches[0]
            if class_res[1] == "SG":
                feats.append("Number=Sing")
            else:
                feats.append("Number=Plur")
                
        bare_class = ""
        if not self.__cls:
            return feats
        
        if "/" not in self.__cls[0]:
            bare_class = self.__cls[0]
        elif "Number=Sing" in feats:     
            bare_class = self.__cls[0].split("/")[0]
        else:
            bare_class = self.__cls[0].split("/")[1]
            
        feats.append("NounClass=Bantu" + bare_class)
        return feats
    
    def get_verb_subj_ud_morph_feats(self) -> list:
        """
        Get the verbal features pertaining to subject information
        """
        features = []
        # the subject person and bantu class values are not marked
        # as anything special, this is done for the purpose of compatibility
        # with existing treebanks which show subject aggreement in this way
        subj = ""
        if self.__subj:
            subj = self.__subj
            
        if "SG1" in subj:
            features.append("Person=1")
            features.append("NounClass=Bantu1")
            features.append("Number=Sing")
        elif "PL1" in subj:
            features.append("Person=1")
            features.append("NounClass=Bantu2")
            features.append("Number=Plur")
        elif "SG2" in subj:
            features.append("Person=2")
            features.append("NounClass=Bantu1")
            features.append("Number=Sing")
        elif "PL2" in subj:
            features.append("Person=2")
            features.append("NounClass=Bantu2")
            features.append("Number=Plur")
        elif "SG3" in subj:
            features.append("NounClass=Bantu1")
            features.append("Person=3")
            features.append("Number=Sing")
        elif "PL3" in subj :
            features.append("NounClass=Bantu2")
            features.append("Person=3")
            features.append("Number=Plur")
        elif "HABIT-SG" in subj:
            features.append("Number=Sing")
        elif "HABIT-PL" in subj:
            features.append("Number=Plur")
        elif subj:
            nc = ""
            num = ""
            try:
                nc, num = subj.split("=")[1].split("-")
            except ValueError:
                warnings.warn(subj + " not possible")
            # make sure subject was specified
            features.append("NounClass=Bantu" + nc)
            if num == "SG":
                features.append("Number=Sing")
            elif num == "PL":
                features.append("Number=Plur")
            features.append("Person=3")
        
        # noun classes all are 3rd person
        return features
    
    def get_verb_obj_ud_morph_feats(self) -> list:
        """
        Get the verbal features pertaining to subject information
        """
        features = []
        # the subject person and bantu class values are not marked
        # as anything special, this is done for the purpose of compatibility
        # with existing treebanks which show subject aggreement in this way
        obj = ""
        if self.__obj:
            obj = self.__obj
            
        obj_pref = "Obj"
        if "SG1" in obj:
            features.append(obj_pref + "Person=1")
            features.append(obj_pref + "NounClass=Bantu1")
            features.append(obj_pref + "Number=Sing")
        elif "PL1" in obj:
            features.append(obj_pref + "Person=1")
            features.append(obj_pref + "NounClass=Bantu2")
            features.append(obj_pref + "Number=Plur")
        elif "SG2" in obj:
            features.append(obj_pref + "Person=2")
            features.append(obj_pref + "NounClass=Bantu1")
            features.append(obj_pref + "Number=Sing")
        elif "PL2" in obj:
            features.append(obj_pref + "Person=2")
            features.append(obj_pref + "NounClass=Bantu2")
            features.append(obj_pref + "Number=Plur")
        elif "SG3" in obj:
            features.append(obj_pref + "NounClass=Bantu1")
            features.append(obj_pref + "Person=3")
            features.append(obj_pref + "Number=Sing")
        elif "PL3" in obj:
            features.append(obj_pref + "NounClass=Bantu2")
            features.append(obj_pref + "Person=3")
            features.append(obj_pref + "Number=Plur")
        elif "HABIT-SG" in obj:
            features.append(obj_pref + "Number=Sing")
        elif "HABIT-PL" in obj:
            features.append(obj_pref + "Number=Plur")
        elif obj:
            try:
                nc, num = obj.split("=")[1].split("-")
                # make sure subject was specified
                features.append(obj_pref + "NounClass=Bantu" + nc)
            except ValueError:
                warnings.warn(obj + " not possible")
            if num == "SG":
                features.append(obj_pref + "Number=Sing")
            elif num == "PL":
                features.append(obj_pref + "Number=Plur")
            features.append(obj_pref + "Person=3")
        
        # noun classes all are 3rd person
        return features
    
    def get_verb_rel_ud_morph_feats(self) -> list:
        """
        Get the verbal features pertaining to relative marker information.
        
        Currently subject relatives are not handled. I don't really know what
        a good way to deal with these is.
        """
        features = []
        # the subject person and bantu class values are not marked
        # as anything special, this is done for the purpose of compatibility
        # with existing treebanks which show subject aggreement in this way
        rel = ""
        if self.__rel:
            rel = self.__rel
            
        print(rel)
        rel_pref = "Rel"
        if "SG1" in rel:
            features.append(rel_pref + "Person=1")
            features.append(rel_pref + "NounClass=Bantu1")
            features.append(rel_pref + "Number=Sing")
        elif "PL1" in rel:
            features.append(rel_pref + "Person=1")
            features.append(rel_pref + "NounClass=Bantu2")
            features.append(rel_pref + "Number=Plur")
        elif "SG2" in rel:
            features.append(rel_pref + "Person=2")
            features.append(rel_pref + "NounClass=Bantu1")
            features.append(rel_pref + "Number=Sing")
        elif "PL2" in rel:
            features.append(rel_pref + "Person=2")
            features.append(rel_pref + "NounClass=Bantu2")
            features.append(rel_pref + "Number=Plur")
        elif "SG3" in rel:
            features.append(rel_pref + "NounClass=Bantu1")
            features.append(rel_pref + "Person=3")
            features.append(rel_pref + "Number=Sing")
        elif "PL3" in rel:
            features.append(rel_pref + "NounClass=Bantu2")
            features.append(rel_pref + "Person=3")
            features.append(rel_pref + "Number=Plur")
        elif "HABIT-SG" in rel:
            features.append(rel_pref + "Number=Sing")
        elif "HABIT-PL" in rel:
            features.append(rel_pref + "Number=Plur")
        elif rel:
            nc = ""
            num = ""
            try:
                nc, num = rel.split("=")[1].split("-")
                # make sure subject was specified
                features.append(rel_pref + "NounClass=Bantu" + nc)
            except ValueError:
                warnings.warn(rel + " not possible")
            if num == "SG":
                features.append(rel_pref + "Number=Sing")
            elif num == "PL":
                features.append(rel_pref + "Number=Plur")
            features.append(rel_pref + "Person=3")
        
        # noun classes all are 3rd person
        return features
    
    def get_verb_ud_morph_feats(self) -> list:
        """
        Get ud morphological features for verbs.
        """
        features = []
        split_msd = self.__msd_extra.split()
        tam = []
        if self.__tam:
            tam = self.__tam
        
        # transitivity checks
        if "SVOO" in split_msd:
            features.append("Subcat=Ditran")
        elif 'SVO' in split_msd:
            features.append("Subcat=Tran")
        elif 'SV' in split_msd:
            features.append("Subcat=Intr")
        
        # get voice stuff, these are not mutually exclusive
        if "APPL" in split_msd:
            features.append("Voice=Appl")
        if "CAUS" in split_msd:
            features.append("Voice=Cau")
        if "PASS" in split_msd or "PS" in split_msd:
            features.append("Voice=Pass")
        else:
            features.append("Voice=Act")
        if "REC" in split_msd:
            features.append("Voice=Rcp")
            
        # handling mood and aspect stuff
        if "PERF" in tam:
            features.append("Aspect=Perf")
        else:
            features.append("Aspect=Imp")
            
        if "SBJN" in tam:
            features.append("Mood=Sub")
        elif "IMP" in tam:
            features.append("Mood=Imp")
        else:
            features.append("Mood=Ind")
            
        if "TAM=COND:nge" in self.__msd or "TAM=COND:ki" in self.__msd:
            features.append("Mood=Cnd")
            features.append("Tense=Pres")
        elif "TAM=COND:ngeli" in self.__msd:
            features.append("Mood=Cnd")
            features.append("Tense=Past")
            features.append("Polarity=Pos")
        elif "TAM=COND:singe" in self.__msd:
            features.append("Mood=Cnd")
            features.append("Tense=Pres")
            features.append("Polarity=Neg")
        # handling tense (that hasn't already been handled
        elif "TAM=PAST-NEG" in self.__msd: # not sure if I'm grabbing this correctly into tense
            features.append("Tense=Past")
            features.append("Polarity=Neg")
        elif "TAM=PAST" in self.__msd:
            features.append("Tense=Past")
            features.append("Polarity=Pos")
        # handling negation
        # this has to happen as part of the previous block because 
        # the singe marker encodes mood tense and polarity
        # and the default polarity pos won't be handled correctly otherwise
        elif "NEG" in tam:
            features.append("Polarity=Neg")
        else:
            features.append("Polarity=Pos")

        if "PR" in tam:
            features.append("Tense=Pres")
        elif "PAST" in tam:
            features.append("Tense=Past")
        elif "FUT" in tam:
            features.append("Tense=Fut")
        # This is actually encoded in the subject not tense
        # because this is where Swahili actually marks this information elsewhere
        if "HABIT-PL" in self.__msd or "HABIT-SG" in self.__msd:
            features.append("Aspect=Hab")
        # stative doesn't seem to belong in the same category 
        # and reduplication definitely does not belong in the voice category
        
        # get subject stuff
        features += self.get_verb_subj_ud_morph_feats()
        features += self.get_verb_obj_ud_morph_feats()
        features += self.get_verb_rel_ud_morph_feats()
        return features
        
            
    def get_adj_ud_morph_feats(self) -> list:
        """
        Get ud morph feats for adjectives.
        """
        features = []
        if "CARD" in self.__msd:
            features.append("NumType=Card")
            
        elif "ORD" in self.__msd:
            features.append("NumType=Ord")
        
        if "COMP" in self.__msd:
            features.append("Degree=Cmp")
        elif "SUPER" in self.__msd:
            features.append("Degree=Sup")
        elif self.__pos == "ADJ":
            features.append("Degree=Pos")
            
        if "NUM-INFL" in self.__msd or "A-INFL" in self.__msd:
            for piece in self.__msd.split():
                match_obj = re.match(r"([0-9]+)\-(SG|PL)", piece)
                if match_obj:
                    features.append("NounClass=Bantu" + str(match_obj.group(1)))
                    num = match_obj.group(2)
                    if num == "PL":
                        features.append("Number=Plur")
                    elif num == "SG":
                        features.append("Number=Sing")
                        
        return features

    def get_fallback_ud_morph_feats(self) -> list:
        """
        This is a blanket fallback to get noun class inflation for all
        other parts of speech
        """
        features = []
        for piece in self.__msd.split():
            match_obj = re.match(r"([0-9]+)\-(SG|PL)", piece)
            if match_obj:
                features.append("NounClass=Bantu" + str(match_obj.group(1)))
                num = match_obj.group(2)
                if num == "PL":
                    features.append("Number=Plur")
                elif num == "SG":
                    features.append("Number=Sing")
                    
        return features
        
    def get_ud_morph_feats(self) -> list:
        """
        Get ud friendly morphological features. 
        
        This has a lot of important information in the comments regarding
        judgement calls that were made on the mapping from Helsinki morphological
        features to universal dependency morphological features.
        """
        
        feats = []
        if self.get_pos() == "NOUN" or self.get_pos() == "N":
            feats += self.get_noun_ud_morph_feats()
        elif self.get_pos() == "VERB" or self.get_pos() == "V":
            feats += self.get_verb_ud_morph_feats()
        elif self.get_pos() == "ADJ" or self.get_pos() == "NUM":
            feats += self.get_adj_ud_morph_feats()
        else:
            feats += self.get_fallback_ud_morph_feats()
        # sort the features alphabetically
        print(feats)
        return "|".join(sorted(feats))
    
    def get_lemma(self):
        """
        Get the lemma from the Helsinki corpus
        """
        return self.__lemma
    
    def set_lemma(self, lemma):
        """
        Set the lemma for a given word
        """
        self.__lemma = lemma
        
    def get_UPOS_tag(self):
        """
        This method goes through a series of steps to change the 
        Helsinki corpus tag to a Universal Part of Speech tag.
        
        Most of the change is direct from one part of speech to another. 
        However, for some, the parts of speech are ambiguous or have poor annotations but
        are easy to guess from the surface form (e.g. punctuation).
        
        For these cases, other parts of the annotated Swahili word must be 
        taken into consideration.
        """
        puncts = [".", "?", ",", "'", "...", ").", ")", "(", ">", '"', ';']
        if self.__word in puncts:
            return "PUNCT"
        
        tag_corr = {
                    "A-UNINFL"  : "ADJ",
                    "ABBR"      : "SYM",   #Probably a good idea to use the syntactic tag for these
                                            #All of the examples I looked at were ADVs though
                    "ADJ"       : "ADJ",
                    "ADV"       : "ADV",
                    "AG-PART"   : "ADP",   #This refers to the  use of 'na' to represent oblique
                                            #subjects in passives
                    "CC"        : "CCONJ",
                    "CONJ"      : "SCONJ", #These are all @CS in the data I looked at, it appears 
                                            # that CC is reserved for coordinating conjunctions and
                                            #Everything else falls under here
                    "CONJ/CC"   : "CCONJ",
                    "DEM"       : "DET",
                    "EXCLAM"    : "INTJ",  #need an exception here because ! is also EXLCLAM
                    "GEN-CON"   : "ADP",   #perhaps a feature is needed to differentiate these?
                    "GEN-CON-KWA": "ADP",
                    "INTERROG"  : "PRON",
                    "N"         : "NOUN",
                    "NUM"       : "NUM",
                    "NUM-ROM"   : "NUM",
                    "POSS-PRON" : "SCONJ", #These are the mwenye, penye, zenye style relative markers
                                            #meaning something like 'which has'
                    "PREP"      : "ADP",
                    "PREP/ADV"  : "ADV",   #These are actually a false compund
                                            # The hcs counts these as a single word but they have a space intervening
                    "PRON"      : "PRON",
                    "PROPNAME"  : "PROPN",
                    "REL-LI"    : "PRON",  #UD treats inflected relative clauses as pronouns 
                    "REL-LI-VYO": "PRON",  # Not subordinating conjunctions
                    "REL-SI"    : "PRON",
                    "REL-SI-VYO": "PRON",
                    "TITLE"     : "PROPN",
                    "V"         : "VERB",
                    "V-BE"      : "AUX",
                    "V-DEF"     : "VERB",  # These so-called 'defective verbs' are almost all
                                            # verbs of arabic origin that have frozen, uninflected forms 
                    }
        punct_pos = [  "COLON",
                    "COMMA",
                    "DOUBLE-QUOTE",
                    "DOUBLE-QUOTE-CLOSING",
                    "DOUBLE-QUOTE-OPENING",
                    "HYPHEN",
                    "LEFT-PARENTHESIS",
                    "PERCENT-MARK",
                    "QUESTION-MARK",
                    "RIGHT-PARENTHESIS",
                    "SEMI-COLON",
                    "SINGLE-QUOTE",
                    "SINGLE-QUOTE-CLOSING",
                    "SINGLE-QUOTE-OPENING",
                    "SLASH",
                    "EQUAL-MARK",
                    "DOLLAR-SIGN",
                    "STOP"]
        
        
        if self.__word == "!" or self.__pos in punct_pos:
            return "PUNCT"

        elif self.__pos is not None and "REL-LI" in self.__pos:
            if "V" in self.__func:
                return "VERB"
            else:
                return "PRON"
        else:
            try:
                return tag_corr[self.__pos]

            except KeyError:
                print(self.__word + "\t" + self.__pos)
                return False

    def strip_MSD(self):
        """
        This method removes the msd tags that specify individual words as these are
        problematic for prediction due to the sparsity of these tags in the dataset
        """
        self.__msd = re.sub(r"\[.*\]", "", self.__msd)
        self.__msd = self.__msd.strip()
        
    def set_cls(self):
        """
        This sets the `__cls` attribute of the tag by parsing the msd information
        """
        if self.__pos == "N":
            class_matches = re.findall(r"\s*([0-9/]+)-((?:PLSG)|(?:SG)|(?:PL))\s*", self.__msd)
            if class_matches:
                class_res = class_matches[0]
                number = class_res
                if class_res[1] == "SG":
                    if "/" in number:
                        self.__cls = re.split("/", number)[0]
                    else:
                        self.__cls = number
                elif class_res[1] == "PL":
                    if "/" in number:
                        self.__cls = re.split("/", number)[1]
                    else:
                        self.__cls = number
                else:
                    self.__cls = number
                 
    def set_pos(self, pos):
        self.__pos = pos 

    def get_surface(self):
        return self.__word

    def get_func(self):
        return self.__func

    def get_pos(self):
        return self.__pos

    def set_verb_tags(self):
        """
        This function sets the tam, subj, obj and rel attributes of a tag if it is a verb, 
        if not these are left alone
        (For non-verbs, these tags are not relevant)
        """
        if self.__pos == "V":
            tam_match = re.findall(r"TAM=([A-Z]+)(?:-([A-Z]+)){0,1}(?:\:([a-z]+)){0,1}", self.__msd)
            if tam_match:
                self.__tam = tam_match[0]
            
            haystack_pieces = re.split(" " , self.__msd)
            for piece in haystack_pieces:
                matches = re.match(r"SUB-PREF=.*", piece)
                if matches is not None:
                    self.__subj = matches.string
                
                matches = re.match(r"OBJ-PREF=.*", piece)
                if matches is not None:
                    #print("found an obj")
                    self.__obj = matches.string
                    
                matches = re.match(r"REL-PREF=", piece)
                if matches is not None:
                    #print("found a rel")
                    self.__rel = matches.string


class Sentence:
    """
    This is the class that will hold the words in the annotation as well as the 
    meta information about the document where this sentence originated from in the corpus.
    """
    def __init__(self, words=[],  sent_id=0, author : str = "", 
                 year : int = -1, orig_title : str = "", orig_filename : str = ""):
        self.__words = words
        self.__id = sent_id
        self.__author = author
        self.__year = year
        self.__orig_title = orig_title
        self.__orig_filename = orig_filename

    def __str__(self):
        line = ""
        for word in self.__words:
            line += word.get_surface() + " "
            
        return line
        
    def set_words(self, words: list):
        self.__words = words

    def get_words(self):
        return self.__words
    
    def add_word(self, word: str):
        self.__words.append(word)

    def set_pos_tags(self):
        for word in self.__words:
            word.set_pos(word.get_UPOS_tag())
            
    def set_verb_tags(self):
        for word in self.__words:
            word.set_verb_tags()
            
    def set_cls(self):
        for word in self.__words:
            word.set_cls()



    def interlinear_print(self):
        """
        This prints out a nice interlinear gloss style output
        where the annotations occur under the text
        it is assumed that the annotations and text are delineated by space
        """
        max_lengths = []
        split_txt = [word.get_surface() for word in self.__words]
        split_anno = [word.get_pos() for word in self.__words]
        n_entries = len(split_txt)
        if n_entries < len(split_anno):
            n_entries = len(split_anno)

        # first find out the maximum size of the entry at that position in 
        # the interlinear analysis
        for i in range(n_entries):
            if i >= len(split_anno):
                max_lengths.append(len(split_txt[i]))

            elif i >= len(split_txt):
                max_lengths.append(len(split_anno[i]))

            else:
                if len(split_txt[i]) > len(split_anno[i]):
                    max_lengths.append(len(split_txt[i]))
                else:
                    max_lengths.append(len(split_anno[i]))

        format_string_txt = ""
        format_string_anno = ""
        for i in range(len(split_txt)):
            format_string_txt += "{:<" + str(max_lengths[i] + 2) + "}"

        for i in range(len(split_anno)):
            format_string_anno += "{:<" + str(max_lengths[i] + 2) + "}"

        print(format_splat(format_string_txt, *split_txt))
        print(format_splat(format_string_anno, *split_anno))
        print()
        print("-" * 40)
        print()

    def identify_nones(self):
        """
        Just trying to find out why so many pos tags are None
        """
        for word in self.__words:
            if word.get_pos() is None:
                print(word.get_surface() + "\t" + word.get_pos())

    def strip_msd(self):
        for word in self.__words:
            word.strip_msd()

class Document:
    """
    This is represented as a sequence of documents.
    This is how individual json files can be read in.
    """
    def __init__(self, sents: list = []):
        self.__sents = sents
        
    def get_sents(self):
        return self.__sents
    
    def add_sent(self, sent:list):
        self.__sents.append(sent)

    def set_upos_tags(self):
        self.remove_missing()
        for sent in self.__sents:
            sent.set_pos_tags()
            
    def set_verb_tags(self):
        for sent in self.__sents:
            sent.set_verb_tags()
            
    def set_noun_classes(self):
        for sent in self.__sents:
            sent.set_cls()
        
    def strip_msd_tags(self):
        for sent in self.__sents:
            sent.strip_msd()
            
    def pop_sent(self, index : int):
        """
        Remove the sentence at the specified index
        """
        self.__sents.pop(index)

    def remove_missing(self):
        """
        Remove all sentences that have words without Part of Speech tags
        """
        sents = self.get_sents()
        new_sents = []
        none_flag = False
        for sent in sents:
            none_flag = False
            for word in sent.get_words():
                if word.get_pos() == False:
                    none_flag = True
                    break
            if not none_flag:
                new_sents.append(sent)
        self.__sents = new_sents

            
    def to_seq_tag_format(self, path : str, label='pos'):
        """
        This outputs the tags to a format that allennlp 
        has dataset readers ready to process. 
        
        This format has each sentence on a single line.
        Each word is followed by `###` and then the label for that word. 
        Each word/tag pair is separated by a tab.
        
        e.g. 
          `WORD###TAG [TAB] WORD###TAG [TAB]`
        """
        if label not in ["pos", "func", "morph", "ud-morph"]:
            print("Incorrect label specified for seq_tag_format write out")
            exit()

        delim = "###"
        with open(path, 'w') as out_file:
            for sent in self.get_sents():
                line = ""
                for word in sent.get_words():
                    line += word.get_surface()
                    line += delim
                    if label == 'pos':
                        line += word.get_pos()
                    elif label == 'func':
                        line += word.get_func()
                    elif label == 'morph':
                        line += word.get_pos() + ":" + word.get_morph_feats()
                    elif label == 'ud-morph':
                        line += word.get_ud_morph_feats()
                    line += '\t'
                out_file.write(line + '\n')
            
            
    def from_json(self, filepath: str):
        """
        This method does the main processing of json files from the helsinki corpus.
        
        It is largely adopted from the julia version of the code. 
        """
        content = ""
        with open(filepath, "r") as in_json:
            content = in_json.read()

        data = json.loads(content)['kwic']
        for hit in data:
            # create a sent
            meta_info = hit['structs']
            year = meta_info['text_year']
            author = meta_info['text_year']
            sent_id = meta_info['sentence_id']
            title = meta_info['text_title']
            filename = meta_info['text_filename']
            # specifying the words=[] is absolutely essential to make sure that the 
            # sentence keeps getting made anew. I have non idea why
            sent = Sentence(words=[], author=author, sent_id=sent_id, year=year, orig_title=title, orig_filename=filename)
            for word in hit['tokens']:
                surface = word['word']
                pos = word['pos']
                func = word['syntax']
                msd = word['msd']
                msd_extra = word["msdextra"]
                lemma = word['lemma']
                gloss = word['gloss']
                new_word = SwahiliTag(surface, pos, msd, gloss=gloss, func=func, lemma=lemma, msd_extra=msd_extra)
                sent.add_word(new_word)
            self.add_sent(sent)
            sent = []
