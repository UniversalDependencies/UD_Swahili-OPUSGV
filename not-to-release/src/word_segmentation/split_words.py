from typing import List
import re

def separate_cc_compounds(input_word: str) -> str:
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
    return re.sub(regex, r"\1 \2", input_word)
