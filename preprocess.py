"""
Text preprocessing.

The sentence is converted into lowercase tokens.
Punctuation is removed.
"""

import re

def tokenize(text):
    text = text.lower()
    return re.findall(r"[a-z]+", text)
