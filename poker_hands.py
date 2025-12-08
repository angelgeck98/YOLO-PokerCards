"""
Optimized Poker Hand Classification Logic
Uses a single-pass analysis for maximum efficiency.
"""

from collections import Counter
from typing import List, Tuple, Dict


# Pre-computed lookup table for rank values
RANK_VALUES = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
    '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
}


class HandAnalysis:
    """
    Analyzes a poker hand in a single pass for efficiency.
    All properties are computed once and cached.
    """
    
    def __init__(self, cards: List[Tuple[str, str]]):
        self.cards = cards
        self.ranks = [rank for rank, suit in cards]
        self.suits = [suit for rank, suit in cards]
        self.values = sorted([RANK_VALUES[rank] for rank in self.ranks])
        self.rank_counts = Counter(self.ranks)
        self.count_values = sorted(self.rank_counts.values(), reverse=True)
        
        # Compute all flags in one pass
        self._is_flush = len(set(self.suits)) == 1
        self._is_straight = self._check_straight()
    
    def _check_straight(self) -> bool:
        """Check if values form a sequence."""
        # Regular straight
        if self.values == list(range(self.values[0], self.values[0] + 5)):
            return True
        # Ace-low straight (A-2-3-4-5)
        if self.values == [2, 3, 4, 5, 14]:
            return True
        return False
    
    @property
    def is_flush(self) -> bool:
        return self._is_flush
    
    @property
    def is_straight(self) -> bool:
        return self._is_straight
    
    @property
    def is_royal(self) -> bool:
        return self.values == [10, 11, 12, 13, 14]
    
    def matches_pattern(self, pattern: List[int]) -> bool:
        """Check if rank counts match a pattern (e.g., [4, 1] for four of a kind)."""
        return self.count_values == pattern


# Lookup table for hand rankings (order matters - check from strongest to weakest)
HAND_RANKINGS = [
    ("Royal Flush", lambda h: h.is_flush and h.is_straight and h.is_royal),
    ("Straight Flush", lambda h: h.is_flush and h.is_straight),
    ("Four of a Kind", lambda h: h.matches_pattern([4, 1])),
    ("Full House", lambda h: h.matches_pattern([3, 2])),
    ("Flush", lambda h: h.is_flush),
    ("Straight", lambda h: h.is_straight),
    ("Three of a Kind", lambda h: h.matches_pattern([3, 1, 1])),
    ("Two Pair", lambda h: h.matches_pattern([2, 2, 1])),
    ("One Pair", lambda h: h.matches_pattern([2, 1, 1, 1])),
    ("High Card", lambda h: True),  # Always matches
]


def classify_hand(cards: List[Tuple[str, str]]) -> str:
    """
    Classify a 5-card poker hand (OPTIMIZED VERSION).
    
    This version uses:
    - Single-pass analysis (compute everything once)
    - Lookup table for hand rankings
    - Pattern matching for rank counts
    
    Args:
        cards: List of (rank, suit) tuples, e.g., [('K', 'S'), ('K', 'H'), ...]
    
    Returns:
        String describing the poker hand rank
    
    Time Complexity: O(n log n) where n=5 (constant time in practice)
    
    Examples:
        >>> classify_hand([('K', 'S'), ('K', 'H'), ('7', 'C'), ('7', 'D'), ('2', 'S')])
        'Two Pair'
        
        >>> classify_hand([('A', 'H'), ('K', 'H'), ('Q', 'H'), ('J', 'H'), ('10', 'H')])
        'Royal Flush'
    """
    if len(cards) != 5:
        return "Invalid hand (must have exactly 5 cards)"
    
    # Single-pass analysis
    analysis = HandAnalysis(cards)
    
    # Check rankings using lookup table
    for hand_name, check_func in HAND_RANKINGS:
        if check_func(analysis):
            return hand_name
    
    return "High Card"  # Fallback (should never reach here)


def classify_hand_with_details(cards: List[Tuple[str, str]]) -> Dict[str, any]:
    """
    Classify hand and return detailed breakdown.
    
    Args:
        cards: List of (rank, suit) tuples
    
    Returns:
        Dictionary with hand classification and analysis details
    """
    if len(cards) != 5:
        return {"hand": "Invalid", "error": "Must have exactly 5 cards"}
    
    analysis = HandAnalysis(cards)
    
    # Find the hand type
    hand_type = None
    for hand_name, check_func in HAND_RANKINGS:
        if check_func(analysis):
            hand_type = hand_name
            break
    
    # Get high card
    high_card_value = max(analysis.values)
    high_card = [rank for rank in analysis.ranks if RANK_VALUES[rank] == high_card_value][0]
    
    return {
        "hand": hand_type,
        "high_card": high_card,
        "is_flush": analysis.is_flush,
        "is_straight": analysis.is_straight,
        "rank_counts": dict(analysis.rank_counts),
        "cards": cards
    }


def get_high_card(cards: List[Tuple[str, str]]) -> str:
    """
    Get the highest card in the hand.
    
    Args:
        cards: List of (rank, suit) tuples
    
    Returns:
        String representing the highest card rank
    """
    ranks = [rank for rank, suit in cards]
    return max(ranks, key=lambda r: RANK_VALUES[r])


# Batch processing for multiple hands (useful for evaluation)
def classify_multiple_hands(hands: List[List[Tuple[str, str]]]) -> List[str]:
    """
    Classify multiple poker hands efficiently.
    
    Args:
        hands: List of hands, where each hand is a list of (rank, suit) tuples
    
    Returns:
        List of hand classifications
    """
    return [classify_hand(hand) for hand in hands]


# Test cases
if __name__ == "__main__":
    print("Testing OPTIMIZED Poker Hand Classifier\n" + "="*50)
    
    test_hands = [
        # Royal Flush
        ([('A', 'H'), ('K', 'H'), ('Q', 'H'), ('J', 'H'), ('10', 'H')], "Royal Flush"),
        
        # Straight Flush
        ([('9', 'D'), ('8', 'D'), ('7', 'D'), ('6', 'D'), ('5', 'D')], "Straight Flush"),
        
        # Four of a Kind
        ([('7', 'H'), ('7', 'D'), ('7', 'S'), ('7', 'C'), ('K', 'H')], "Four of a Kind"),
        
        # Full House
        ([('K', 'S'), ('K', 'H'), ('K', 'D'), ('2', 'C'), ('2', 'S')], "Full House"),
        
        # Flush
        ([('K', 'H'), ('10', 'H'), ('7', 'H'), ('5', 'H'), ('2', 'H')], "Flush"),
        
        # Straight
        ([('9', 'H'), ('8', 'D'), ('7', 'S'), ('6', 'C'), ('5', 'H')], "Straight"),
        
        # Three of a Kind
        ([('Q', 'S'), ('Q', 'H'), ('Q', 'D'), ('5', 'C'), ('2', 'S')], "Three of a Kind"),
        
        # Two Pair
        ([('K', 'S'), ('K', 'H'), ('7', 'C'), ('7', 'D'), ('2', 'S')], "Two Pair"),
        
        # One Pair
        ([('A', 'S'), ('A', 'H'), ('9', 'C'), ('6', 'D'), ('3', 'S')], "One Pair"),
        
        # High Card
        ([('K', 'S'), ('Q', 'H'), ('9', 'C'), ('5', 'D'), ('2', 'S')], "High Card"),
        
        # Ace-low straight
        ([('A', 'H'), ('2', 'D'), ('3', 'S'), ('4', 'C'), ('5', 'H')], "Straight"),
    ]
    
    print("Basic Classification Tests:")
    print("-" * 50)
    passed = 0
    for cards, expected in test_hands:
        result = classify_hand(cards)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        print(f"{status} Expected: {expected:20s} Got: {result:20s}")
    
    print(f"\nTests passed: {passed}/{len(test_hands)}")
    
    # Test detailed classification
    print("\n" + "="*50)
    print("Detailed Classification Example:")
    print("-" * 50)
    test_hand = [('K', 'S'), ('K', 'H'), ('7', 'C'), ('7', 'D'), ('2', 'S')]
    details = classify_hand_with_details(test_hand)
    print(f"Cards: {test_hand}")
    print(f"Hand: {details['hand']}")
    print(f"High Card: {details['high_card']}")
    print(f"Is Flush: {details['is_flush']}")
    print(f"Is Straight: {details['is_straight']}")
    print(f"Rank Counts: {details['rank_counts']}")
    
    # Performance test
    print("\n" + "="*50)
    print("Performance Test (1000 hands):")
    print("-" * 50)
    import time
    
    test_batch = [test_hands[0][0] for _ in range(1000)]
    start = time.time()
    results = classify_multiple_hands(test_batch)
    elapsed = time.time() - start
    print(f"Classified {len(results)} hands in {elapsed:.4f} seconds")
    print(f"Average: {elapsed/len(results)*1000:.4f} ms per hand")