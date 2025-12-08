'''
Placeholder file for poker hand evaluation functions
Is this the same as evaluation.py???
'''

from collections import Counter
from typing import List, Tuple, Dict

RANK_VALUES = {

    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
    '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14

}

class HandAnalyis: 
    """
    Analyzes a poker hand in a single pass for efficiency.
    All properties are computed once and cached.
    """

    def __init__(self, cards: List[Tuple[str, str]]):
        self.cards = cards 
        self.ranks = [rank for rank, suit in cards]
        self.suits = [suit for rank, suit in cards]
        self.values = sorted([RANK_VALUES[rank] for rank in self.ranks])