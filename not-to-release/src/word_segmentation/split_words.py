from typing import List, Dict
import re

def check_keys(input_row: Dict) -> bool:
    """
    Check that the appropriate keys are present in the input dictionary. 
    """
    for key in ['word', 'upos', 'pos', 'lemma', 'morph', 'func']:
        if key not in input_row:
            return False
    
    return True

    
def preprocess_cc_compounds(input_word: str) -> str:
    """
    Make compounds into their own separate words.
    
    Params
    -----
    input_word: str
        The
    
    Returns
    -------
    
    """
    # This handles 'naye', 'nayo', nao', nami, nawe, naye, nasi, nanyi, nao
    # nalo, nacho, navyo, nazo, nako, napo, namo
    regex = r'(na)(ye|yo|o|mi|we|si|nyi|lo|cho|vyo|zo|ko|po|mo)'
    match_dict = {'naye' :'na yeye',
                  'nayo' : 'na yo',
                  'nao' : 'na wao',
                  'nami' : 'na mimi',
                  'nawe', 'na wewe',
                  'nasi', 'na sisi',
                  'nanyi', 'na nyinyi',
                  'nacho', 'na hicho',
                  'navyo', 'na hivi',
                  'nako', 'na huko'
                  'namo', 'na humo',
                  'napo', 'na hapo',
                  'nazo', 'na hizi',
                  'nalo', 'na hili'}
    
    
def postprocess_cc_compounds(input_row:dict) -> List[Dict]
    """
    If your model was already trained using the input with cc compounds, you can 
    separate them after the fact using this function. 
    """
    assert check_keys(input_row)
    match_dict = {'naye' : 
                      {'word': 'yeye', 'lemma' : 'yeye', 'upos': 'PRON', }
                'nayo' : 'na yo',
                'nao' : 'na wao',
                'nami' : 'na mimi',
                'nawe', 'na wewe',
                'nasi', 'na sisi',
                'nanyi', 'na nyinyi',
                'nacho', 'na hicho',
                'navyo', 'na hivi',
                'nako', 'na huko'
                'namo', 'na humo',
                'napo', 'na hapo',
                'nazo', 'na hizi',
                'nalo', 'na hili'}
    na_row = {'word': 'na', 'lemma': 'na', 'upos': 'CCONJ', 'pos': 'CC', 'morph' : '_', 'func' : '_'}
    if 
