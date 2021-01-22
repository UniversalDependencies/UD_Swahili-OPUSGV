import sys
sys.path.append("../src")
from word_segmentation import convert_verb_complex
from word_segmentation.convert_verb_complex import split_verb, separate_morph_tags, parse_morph_features
import pytest

def test_split_verb_no_object():
    """
    Test that the verbs can be split correctly using the regexes written.
    """
    input_verb = "anapenda"
    first_half = "ana"
    second_half = "penda"
    res_first_half, res_second_half = split_verb(input_verb)
    assert res_first_half == first_half
    assert res_second_half == second_half
    input_verb = "walipenda"
    first_half = "wali"
    second_half = "penda"
    res_first_half, res_second_half = split_verb(input_verb)
    assert res_first_half == first_half
    assert res_second_half == second_half
    input_verb = "kitapenda"
    first_half = "kita"
    second_half = "penda"
    res_first_half, res_second_half = split_verb(input_verb)
    assert res_first_half == first_half
    assert res_second_half == second_half
    input_verb = "tunapenda"
    first_half = "tuna"
    second_half = "penda"
    res_first_half, res_second_half = split_verb(input_verb)
    assert res_first_half == first_half
    assert res_second_half == second_half
    input_verb = "sipendi"
    first_half = "si"
    second_half = "pendi"
    res_first_half, res_second_half = split_verb(input_verb)
    assert res_first_half == first_half
    assert res_second_half == second_half
    input_verb = "hazipendi"
    first_half = "hazi"
    second_half = "pendi"
    res_first_half, res_second_half = split_verb(input_verb)
    assert res_first_half == first_half
    assert res_second_half == second_half
    
def test_split_verb_w_object():
    """
    Test that the verbs can be split correctly using the regexes written.
    """
    input_verb = "anakipenda"
    first_half = "ana"
    second_half = "kipenda"
    res_first_half, res_second_half = split_verb(input_verb)
    assert res_first_half == first_half
    assert res_second_half == second_half
    input_verb = "walizipenda"
    first_half = "wali"
    second_half = "zipenda"
    res_first_half, res_second_half = split_verb(input_verb)
    assert res_first_half == first_half
    assert res_second_half == second_half
    input_verb = "kitaupenda"
    first_half = "kita"
    second_half = "upenda"
    res_first_half, res_second_half = split_verb(input_verb)
    assert res_first_half == first_half
    assert res_second_half == second_half
    input_verb = "tunaipenda"
    first_half = "tuna"
    second_half = "ipenda"
    res_first_half, res_second_half = split_verb(input_verb)
    assert res_first_half == first_half
    assert res_second_half == second_half
    input_verb = "sikipendi"
    first_half = "si"
    second_half = "kipendi"
    res_first_half, res_second_half = split_verb(input_verb)
    assert res_first_half == first_half
    assert res_second_half == second_half
    input_verb = "hawalivipendi"
    first_half = "hawali"
    second_half = "vipendi"
    res_first_half, res_second_half = split_verb(input_verb)
    assert res_first_half == first_half
    assert res_second_half == second_half

def test_morph_tag_parsing():
    """
    Test that morphological tag parsing is working correctly
    """
    input_morph_string = "NounClass[Subj]=Bantu1|NounClass[Obj]=Bantu3|Tense=Pres"
    target_morph_dict = {"NounClass[Subj]" : "Bantu1",
                         "NounClass[Obj]" : "Bantu3",
                         "Tense" : "Pres"}
    assert parse_morph_features(input_morph_string) == target_morph_dict
