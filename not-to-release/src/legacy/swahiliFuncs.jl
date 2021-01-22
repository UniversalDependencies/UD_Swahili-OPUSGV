mutable struct swahiliTag
    word::String
    pos::String
    msd::String
    behind::String
    ahead::String
    cls::String
    tam::String
    subj::String
    obj::String
    rel::String
    gloss::String
    syntax::String
    lemma::String
end
#Outer constructor for swahiliTag to provide default arguments
swahiliTag(word,
            pos,
            msd;
            behind="-",
            ahead="-",
            cls="-",
            tam="-",
            subj="-",
            obj="-",
            rel="-",
            gloss="-",
            syntax="-",
            lemma="-") = 
        swahiliTag(word, pos, msd, 
                        behind, ahead,
                        cls, tam, subj,
                        obj, rel, gloss, 
                        syntax,lemma)
mutable struct swahiliQuery
    jsonFileName::String
    words::Array{swahiliTag,1}
    data::Dict{String, Any}
    uniques::Array{String, 1}
end

#Outer constructor for swahiliQuery to provide default arguments
swahiliQuery(jsonFileName,
            words=Array{swahiliTag,1}(),
            data=Dict{String, Any}(),
            uniques=Array{String,1}()) =
        swahiliQuery(jsonFileName, words, data, uniques)
        
#type for holding sentence information taken from the helsinki corpus 

"""
This class holds the information for a single sentence in an interlinear text
This is used to extract the tag sequences for nltk's hmm tagger

There are no default arguments for this because both attributes are things that cannot be assigned defaults
"""
mutable struct sentence
    words::Array{swahiliTag, 1}
    gloss::String
end

#Additional swahliQuery type that 


struct dependency
    head::Int 
    dep::Int
    label::String
end
# outer constructor for dependency
dependency(head, dep, label="-") = dependency(head, dep, label)

mutable struct dependencySent
    tokenList::Array{swahiliTag,1}
    depList::Array{dependency, 1}
end

function getDepParse(sent::dependencySent)
    tokenList = sent.tokenList
    deps = sent.dependency
    for tokenI = 1:length(tokenList)
        token = tokenList[tokenI]
        if token.syntax == "@<DN"
            unifyTag = swahiliTag("-", "N", "-")
            nounID = seekTagIndex(sent, tokenI, unifyingTag=unifyTag, direction="Aft")
            push!(deps, dependency(nounID, tokenI)) 
        end
    end
end
function seekTagIndex(sent::Array{swahiliTag,1}, startIndex; unifyTag::swahiliTag=swahiliTag(), direction::String="None")
    i = startIndex
    found = false
    if direction != "None"
        while i >= 1 && i <= length(sent)
            if direction == "Fore"
                i += 1
            else
                i -= 1
            end
        end
    else
        while i + startIndex <= length(sent) && i - startIndex >= 1

        end
    end
end
#function isUnifiable()
#
#end
function getCoNLLU(sent::dependencySent)
    conll = ""
    for tokI = 1:length(sent.tokenList)
        token = sent.tokenList[depI]
        head, label = [(arc.head, arc.label) for arc in sent.depList if arc.dep == tokI]
        form = token.word
        lemma = token.lemma
        upos = getUPosTag(token.pos)
        xpos = token.pos

        # The structure of this is
        # ID FORM LEMMA UPOSTAG XPOSTAG FEATS HEAD DEPREL DEPS MISC
        line = string(tokI) * token.word * token.lemma * token.getUPosTag(token.pos) 
    end
    return conll
end

"""
This function does translation of tags from the Helsinki Corpus of Swahili
to Universal Part of Speech tags
"""
function getUPosTag(tag::swahiliTag)

    tagCorr = Dict(
                    "ABBR"      => "SYM",   #Probably a good idea to use the syntactic tag for these
                                            #All of the examples I looked at were ADVs though
                    "ADJ"       => "ADJ",
                    "ADV"       => "ADV",
                    "AG-PART"   => "ADP",   #This refers to the  use of 'na' to represent oblique
                                            #subjects in passives
                    "CC"        => "CCONJ",
                    "CONJ"      => "SCONJ", #These are all @CS in the data I looked at, it appears 
                                            # that CC is reserved for coordinating conjunctions and
                                            #Everything else falls under here
                    "CONJ/CC"   => "CCONJ",
                    "DEM"       => "DET",
                    "EXCLAM"    => "INTJ",  #need an exception here because ! is also EXLCLAM
                    "GEN-CON"   => "ADP",   #perhaps a feature is needed to differentiate these?
                    "GEN-CON"   => "ADP",
                    "INTERROG"  => "PRON",
                    "N"         => "NOUN",
                    "NUM"       => "NUM",
                    "NUM-ROM"   => "NUM",
                    "POSS-PRON" => "SCONJ", #These are the mwenye, penye, zenye style relative markers
                                            #meaning something like 'which has'
                    "PREP"      => "ADP",
                    "PREP/ADV"  => "ADV",   #These are actually a false compund
                                            # The hcs counts these as a single word but they have a space intervening
                    "PRON"      => "PRON",
                    "PROPNAME"  => "PROPN",
                    "REL-LI"    => "PRON",  #UD treats inflected relative clauses as pronouns 
                    "REL-LI-VYO"=> "PRON",  # Not subordinating conjunctions
                    "REL-SI"    => "PRON",
                    "REL-SI-VYO"=> "PRON",
                    "TITLE"     => "PROPN",
                    "V"         => "VERB",
                    "V-BE"      => "AUX",
                    "V-DEF"     => "VERB",  # These so-called 'defective verbs' are almost all
                                            # verbs of arabic origin that have frozen, uninflected forms 
                    )
    puncts = [  "COLON", 
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
                "STOP"]
    #Catch !
    if tag.word == "!" || tag.pos in puncts
        return "PUNCT"
    elseif contains(tag.pos, "REL-LI")
        if contains(tag.syntax, "V")    #Some of these are functioning as verbs and some are functioning as nouns in sentences
                                        #the only clue is in the syntax piece of the word  
            return "VERB"
        else
            return "PRON"
        end
    else 
        return tagCorr[tag.pos]
    end
end
#function getUDFeats(tag::swahiliTag)
#    """
#    This function uses the information in the tag to determine what the appropriate
#    features using the UD feature inventory are.
#    """
#    break 
#end

"""
This function removes the msd tags that specify individual words, these are too easy to guess
"""
function stripMSD!(tag::swahiliTag)
    tag.msd = replace(tag.msd, r"\[.*\]", "")
    tag.msd = strip(replace(tag.msd, r"\:.*[ ]?", " "))
end

"""
This function sets the cls attribute of a tag if it is a noun, if not it is left in the default
"""
function setCLS!(tag::swahiliTag)
    if tag.pos == "N"
        classSearch = eachmatch(r"\s*([0-9/]+)-((?:PLSG)|(?:SG)|(?:PL))\s*", tag.msd)
        if !isempty(classSearch)
            classRes = start(classSearch).match
            number = classRes
            if classRes[2] == "SG"
                if contains(number, "/")
                    tag.cls = split(number, "/")[1]
                else
                    tag.cls = number
                end
            elseif classRes[2] == "PL"
                if contains(number, "/")
                    tag.cls = split(number, "/")[2]
                else
                    tag.cls = number
                end
            else
                tag.cls = number
            end
        end
    end
end
"""
This function sets the tam, subj, obj and rel attributes of a tag if it is a verb, if not these are left alone
(For non-verbs, these tags are not relevant)
"""
function setVerbTags!(tag::swahiliTag)
    if tag.pos == "V"
        classSearch = eachmatch(r"TAM=([A-Z]+)(?:-([A-Z]+)){0,1}(?:\:([a-z]+)){0,1}", tag.msd)
        if !isempty(classSearch)
            tag.tam = start(classSearch).match
        end
        haystackPieces = split(tag.msd)
        for piece in haystackPieces
            index = search(piece, r"SUBJ-PREF=.*")
            if index != 0:-1
                tag.subj = piece[index]
            end
            index = search(piece, r"OBJ-PREF=.*")
            if index != 0:-1
                tag.obj = piece[index]
            end
            index = search(piece, r"REL-PREF=")
            if index != 0:-1
                tag.rel = piece[index]
            end
        end
    end
end

function setWordContext!(tag::swahiliTag; behind=nothing, ahead=nothing, prefix=true)
    if prefix == true
        if behind != nothing
            if prefix==true && length(behind) > 2
                tag.behind = lowercase(behind)[1:2]
            else
                tag.behind = lowercase(behind)
            end
        else
            tag.behind = "nothingBOS"
        end
        if ahead != nothing
            if prefix == true && length(ahead) > 2
                tag.ahead = lowercase(ahead)[1:2]
            else
                tag.ahead = lowercase(ahead)
            end
        else
            tag.ahead = "nothingEOS"
        end
    end
end

function getCharNGrams(tag::swahiliTag, n::Int)
    grams = []
    i = 1
    #Add buffer
    word = " " * tag.word * " "^n 
    for i in eachindex(word[1:end-n])
        push!(grams, word[i:i+n-1])
    end
    return grams
end

"""
    getFeatureValues(swahiliTag, uniques)

uniques must be an array of strings. Fill in a dictionary with the count or simple boolean values

"""
function getFeatureValues(tag::swahiliTag, uniques::Array{String}, counts=true)
    #initialize the dictionary
    initData = zeros(Int, length(uniques))
    featureDict = Dict(zip(uniques, initData))
    for n in [1,2,3,4]
        for gram in getCharNGrams(tag, n)
            try
                featureDict[gram] = counts ? featureDict[gram] + 1 : 1
            end
        end
    end
    if word.behind != "nothingBOS"
        try
            featureDict[word.behind] = counts ? featureDict[word.behind] + 1 : 1
        end
    end
    if word.ahead != "nothingEOS"
        try
            featureDict[word.ahead] = counts ? featureDict[word.ahead] + 1 : 1
        end
    end
    return featureDict
end
"""
    getFeatureDict(swahiliTag)

Fill in a dictionary with the count or simple boolean values

"""
function getFeatureValues(word::swahiliTag, counts=true)
    #initialize the dictionary
    featureDict = Dict{String, Int}()
    for n in [1,2,3,4]
        for gram in getCharNGrams(word, n)
            try
                featureDict[gram] = counts ? featureDict[gram] + 1 : 1
            catch err
                if isa(err, KeyError)
                    featureDict[gram] = 1
                end
            end
        end
    end
    if word.behind != "nothingBOS"
        try
            featureDict[word.behind] = counts ? featureDict[word.behind] + 1 : 1
        catch err
            if isa(err, KeyError)
                featureDict[word.behind] = 1
            end
        end
    end
    if word.ahead != "nothingEOS"
        try
            featureDict[word.ahead] = counts ? featureDict[word.ahead] + 1 : 1
        catch err
            if isa(err, KeyError)
                featureDict[word.ahead] = 1
            end
        end
    end
    return featureDict
end

"""
    fillQuery!(query)

Fill the query's words and unique attributes using the json file in
`swahiliQuery.jsonFileName`

cases where the filename is actually a directory are handled appropriately.
"""
function fillQuery!(query::swahiliQuery)
    uniques = []
    sents = []
    if isdir(query.jsonFileName)
        files = readdir(query.jsonFileName)
        res = @sync @parallel vcat for file in files
            tempUniques = []
            println("Processing file " * file)
            words = getWordsFromJSON(query.jsonFileName * file)
            sents = vcat(sents, words)
            for word in words
                append!(tempUniques, keys(getFeatureValues(word)))
            end
            (tempUniques, sents)
        end
        tempUniques, tempWords = collect(zip(res...))
        #query.uniques = unique(uniques)
        query.uniques = unique(tempUniques[1])#[result[1] for result in res]
        query.words = tempWords[1]#[result[2] for result in res]
        println(length(query.uniques))
        println(length(query.words))
    else
        query.words = getWordsFromJSON(query.jsonFileName)
        uniques = []
        for word in query.words
            append!(uniques,keys(getFeatureValues(word)))
        end
        query.uniques = unique(uniques)
    end
end

"""
    getWordsFromJSON(filepath)

Generate a list of swahiliTag objects for the words that occur in the specified json filepath

"""    
function getWordsFromJSON(filepath::String, pullUniques=true)
    data = JSON.parsefile(filepath)["kwic"]
    words = Array{swahiliTag, 1}()
    for hit in data
        hitSize = length(hit["tokens"])
        for tokenI = 1:hitSize
            token = hit["tokens"][tokenI]
            if true #token["pos"] in ["V","N"] Uncomment this to filter pos tags
                #try
                    token = hit["tokens"][tokenI]
                    for key in keys(token)
                        if token[key] == nothing
                            token[key] = "-"
                        end
                    end

                    currentTag = swahiliTag(token["word"],
                                                token["pos"],
                                                token["msd"],
                                                gloss=token["gloss"],
                                                syntax=token["syntax"]
                                                )
                    previousToken = tokenI != 1 ? hit["tokens"][tokenI - 1]["word"] : nothing 
                    nextToken = tokenI != hitSize ? hit["tokens"][tokenI + 1]["word"] : nothing
                    stripMSD!(currentTag)
                    setCLS!(currentTag)
                    setVerbTags!(currentTag)
                    setWordContext!(currentTag, behind=previousToken, ahead=nextToken)
                    push!(words, currentTag)
                #catch a 
                #    print(a)
                #    println("token")
                #    println(token["word"])
                #    println(token["pos"])
                #    println(token["msd"])
                #    println(token["gloss"])
                #    println(token["syntax"])
                #end
            end
        end
    end
    return words
end
"""
    getSentsFromJSON(filepath)

Generate a list of sentences. 
Each sentence is itself a list of swahiliTag objects for the words that occur in the specified json filepath

"""  
function getSentsFromJSON(filepath::String)
    data = JSON.parsefile(filepath)["kwic"]
    sents = Array{Array{swahiliTag,1},1}()
    for hit in data
        words = Array{swahiliTag, 1}()
        hitSize = length(hit["tokens"])
        for tokenI = 1:hitSize
            token = hit["tokens"][tokenI]
            try
                token = hit["tokens"][tokenI]
                for key in keys(token)
                    if token[key] == nothing
                        token[key] = "-"
                    end
                end
                currentTag = swahiliTag(token["word"],
                                            token["pos"],
                                            token["msd"],
                                            gloss=token["gloss"],
                                            syntax=token["syntax"]
                                            )

                previousToken = tokenI != 1 ? hit["tokens"][tokenI - 1]["word"] : nothing 
                nextToken = tokenI != hitSize ? hit["tokens"][tokenI + 1]["word"] : nothing
                stripMSD!(currentTag)
                setCLS!(currentTag)
                setVerbTags!(currentTag)
                push!(words, currentTag)
            catch err
                println("ERROR" * err.msg)
                println("token")
                println(token["word"])
                println(token["msd"])
                println(token["gloss"])
                println(token["syntax"])
                exit()
            end
        end
        push!(sents, words)
    end
    return sents
end

"""
This function converts sents into a flattened swahiliQuery objects
"""
function flattenSents(sents)
    #Define a new swahili query object 
    output = swahiliQuery("-")
    words = []
    for sent in sents
        for word in sent
            push!(words, word)
        end
    end
end

function extractSents(filepath)
    sents = []
    if isdir(filepath)
        files = readdir(filepath)
        sents = @sync @parallel append! for file in files
            print("\r" * " "^50)
            print("\rProcessing " * filepath * file)
            getSentsFromJSON(filepath * file)
        end
    else
        sents = getSentsFromJSON(filepath)
    end
    println()
    return sents
end
#::Array{Array{swahiliTag, 1},1}
function convertSentsToHmm(sents)
    mutatedSents = @sync @parallel vcat for sent in sents
        toHmmTuple(sent)
    end
    return mutatedSents
end

function toHmmTuple(sent)
    hmmTuple = []#Array{Tuple{String, String}, 1}
    for word in sent
        push!(hmmTuple, (word.word, word.pos))
    end
    return [hmmTuple]
end

#function writeCONLL(query:: )
function createMatrix(query::swahiliQuery, cols::Array)
    println("There are " * string(length(cols)) * " features being extracted")
    println("There are " * string(length(query.words)) * " words to gather features for")
    words = query.words
    @inbounds classes = [word.cls for word in words]
    @inbounds pos = [word.pos for word in words]
    @inbounds msd = [word.msd for word in words]
    @inbounds obj = [word.obj for word in words]
    @inbounds subj = [word.subj for word in words]
    @inbounds tam = [word.tam for word in words]
    @inbounds rel = [word.rel for word in words]

    println("Current Progress")
    dataArray = []
    for wordI = 1:length(query.words)
        if wordI % 100 == 0
            println(string(wordI))
        end
        featsForExample = getFeatureValues(words[wordI])
        @inbounds tempRow = SparseArrays.sparsevec([feat in keys(featsForExample) ? featsForExample[feat] : 0 for feat in cols])
        #[tempRow]
        push!(dataArray, tempRow)
    end
    return Dict("classes"=>classes, 
                "features"=>cols,
                "data"=>dataArray,
                "pos"=>pos, 
                "msd"=>msd, 
                "obj"=>obj,
                "subj"=>subj,
                "tam"=>tam,
                "rel"=>rel)
end

function sampleMatrix(n_samples, matrix)
    sampledTrain = Dict(key => sample(n_samples, matrix[key]) for key in keys(trainData))
    return sampledTrain
end

function sample(n_samples, iterables...)
    length_iter = length(iterables[1])
    #Check that iterables all match in length
    for iterable in iterables
        if length(iterable) != length_iter
            jl_error("Iterables differ in length")
            jl_error("Target Length: " * str(length_iter))
            jl_error("Current Iterable Length: " * str(length(iterable)))
        end
    end
    srand(32109)
    indexes = rand(1:length_iter, n_samples)
    new_iterables = []
    for iterable in iterables
        sampled_iter = [iterable[ind] for ind in indexes]
        push!(new_iterables, sampled_iter)
    end
    return new_iterables
end

function writeData(data::Dict, path::String, key::String="pos")
    println("Writing labels")
    outLabels = open(path * ".pkl", "w")
    pickle.dump(Dict(key => data[key] for key in keys(data) if key != "data"), outLabels)
    close(outLabels)
    println("Writing data")
    outData = open(path, "w")
    for row in data["data"]
        fullRow = full(row)
        for cell in fullRow[1:end-1]
            write(outData, string(cell) * "\t")
        end
        write(outData, string(fullRow[end]) * "\n")
    end
    close(outData)
end

function writePkl(data, path::String)
    out = open(path, "w")
    pickle.dump(data, out)    
end
