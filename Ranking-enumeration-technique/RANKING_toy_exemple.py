"""
Toy script to explore and understand partition block ranking.
Note: ranks are 1-indexed, but the base formula is 0-indexed,
hence the small correction for col_result and line_result.
A clean version should have been 0-indexed (as per the theory) and without correction,
but due to lack of time, this version still works perfectly as is.

This is basically the setup used in experiments:
    The returned indices are then processed by a helper that accesses the column and line matrices for the given rank.
    The product of initial_point * col_matrice * line_matrice yields the requested TSP solution as determined by the group ranking.
"""

import numpy as np
import math
from Cn_representation import Cn_representation

_cached_representations_1 = []
_cached_representations_1 = []
_cached_representations_2 = []
_cached_representations_3 = []
_cached_representations_4 = []
_cached_representations_5 = []

def initialize_representations_1(sizes, truncated):
    """Initialize the cached representations for all required sizes."""
    global _cached_representations_1
    for i in range(len(sizes)):
        _cached_representations_1.append(Cn_representation(sizes[i],truncated[i]))

def initialize_representations_2(sizes, truncated):
    """Initialize the cached representations for all required sizes."""
    global _cached_representations_2
    for i in range(len(sizes)):
        _cached_representations_2.append(Cn_representation(sizes[i],truncated[i]))

def initialize_representations_3(sizes, truncated):
    """Initialize the cached representations for all required sizes."""
    global _cached_representations_3
    for i in range(len(sizes)):
        _cached_representations_3.append(Cn_representation(sizes[i],truncated[i]))

def initialize_representations_4(sizes, truncated):
    """Initialize the cached representations for all required sizes."""
    global _cached_representations_4
    for i in range(len(sizes)):
        _cached_representations_4.append(Cn_representation(sizes[i],truncated[i]))

def initialize_representations_5(sizes, truncated):
    """Initialize the cached representations for all required sizes."""
    global _cached_representations_5
    for i in range(len(sizes)):
        _cached_representations_5.append(Cn_representation(sizes[i],truncated[i]))

def reverse_cached_representation(cached_representation):
    """Reverse the values of a cached representation dictionary."""
    max_value = max(cached_representation.values())
    reversed_representation = {key: max_value - value + 1 for key, value in cached_representation.items()}
    return reversed_representation

#384
format_1 = [[3,4], [2,7]]  # Number of bits [col, line]
sizes_1 = [3,4,2,7]  # Sizes needed for Cn representations
truncated_1=[True,True,True,False] # Truncated or not

#64
format_2 = [[1,2,3,4], [6]]  # Number of bits [col, line]
sizes_2 = [1, 2, 3, 4, 6]  # Sizes needed for Cn representations
truncated_2=[False,True,True,True,False] # Truncated or not

#192
format_3 = [[1,3,4], [6,2]]  # Number of bits [col, line]
sizes_3 = [1 ,3 ,4 ,6 ,2]  # Sizes needed for Cn representations
truncated_3=[False,True,True,False,True] # Truncated or not

#128
format_4 = [[2,3,4],[7]]  # Number of bits [col, line]
sizes_4 = [2,3,4,7]  # Sizes needed for Cn representations
truncated_4= [True,True,True,False] # Truncated or not

#720
format_5 = [[3,3],[4,4,2]]  # Number of bits [col, line]
sizes_5 = [3,3,4,4,2]
truncated_5 = [True, False, True, False, True]


initialize_representations_1(sizes_1, truncated_1)
initialize_representations_2(sizes_2, truncated_2)
initialize_representations_3(sizes_3, truncated_3)
initialize_representations_4(sizes_4, truncated_4)
initialize_representations_5(sizes_5, truncated_5)

def group_384(G, s: str, format=format_1) -> np.array:

    start = 0
    i = 0
    col_result = 0
    line_result = 0
    n_col = 105
    n_line = 384

    reverse_next = False  # Flag to determine if the next cached representation should be reversed

    # Calculate for columns
    mod = 1
    for col_bits in format[0]:
        bits = int(col_bits)
        chunk = s[start:start + bits]
        current_representation = _cached_representations_1[i]
        if reverse_next:
            current_representation = reverse_cached_representation(current_representation)
        rank = current_representation.get(chunk, 1)
        mod *= len(current_representation)
        col_result += rank * (n_col // mod)
        # Check if the rank of the current chunk is 0 mod 2
        reverse_next = (rank % 2 == 0)
        start += bits
        i += 1
    col_result -= 15

    # Calculate for lines
    mod = 1
    for line_bits in format[1]:
        bits = int(line_bits)
        chunk = s[start:start + bits]
        current_representation = _cached_representations_1[i]
        if reverse_next:
            current_representation = reverse_cached_representation(current_representation)
        rank = current_representation.get(chunk, 1)
        mod *= len(current_representation)
        line_result += rank * (n_line // mod)
        # Check if the rank of the current chunk is 0 mod 2
        reverse_next = (rank % 2 == 0)
        start += bits
        i += 1

    line_result -= 128
    return [col_result, line_result]

def group_64(G, s: str, format=format_2) -> np.array:

    start = 0
    i = 0
    col_result = 0
    line_result = 0
    n_col = 630
    n_line = 64

    reverse_next = False

    # Calculate for columns
    mod = 1
    for col_bits in format[0]:
        bits = int(col_bits)
        chunk = s[start:start + bits]
        current_representation = _cached_representations_2[i]
        if reverse_next:
            current_representation = reverse_cached_representation(current_representation)
        rank = current_representation.get(chunk, 1)
        mod *= len(current_representation)

        col_result += rank * (n_col // mod)
        reverse_next = (rank % 2 == 0)
        start += bits
        i += 1
    col_result -= 435

    # Calculate for lines
    mod = 1
    for line_bits in format[1]:
        bits = int(line_bits)
        chunk = s[start:start + bits]
        current_representation = _cached_representations_2[i]
        if reverse_next:
            current_representation = reverse_cached_representation(current_representation)
        rank = current_representation.get(chunk, 1)
        mod *= len(current_representation)
        line_result += rank * (n_line // mod)
        reverse_next = (rank % 2 == 0)
        start += bits
        i += 1

    return [col_result, line_result]

def group_192(G, s: str, format=format_3) -> np.array:

    start = 0
    i = 0
    col_result = 0
    line_result = 0
    n_col = 210
    n_line = 192

    reverse_next = False

    # Calculate for columns
    mod = 1
    for col_bits in format[0]:
        bits = int(col_bits)
        chunk = s[start:start + bits]
        current_representation = _cached_representations_3[i]
        if reverse_next:
            current_representation = reverse_cached_representation(current_representation)
        rank = current_representation.get(chunk, 1)
        mod *= len(current_representation)
        col_result += rank * (n_col // mod)
        reverse_next = (rank % 2 == 0)
        start += bits
        i += 1
    col_result -= 120

    # Calculate for lines
    mod = 1
    for line_bits in format[1]:
        bits = int(line_bits)
        chunk = s[start:start + bits]
        current_representation = _cached_representations_3[i]
        if reverse_next:
            current_representation = reverse_cached_representation(current_representation)
        rank = current_representation.get(chunk, 1)
        mod *= len(current_representation)
        line_result += rank * (n_line // mod)
        reverse_next = (rank % 2 == 0)
        start += bits
        i += 1

    line_result -= 3
    return [col_result, line_result]

def group_128(G, s: str, format=format_4) -> np.array:

    start = 0
    i = 0
    col_result = 0
    line_result = 0
    n_col = 315
    n_line = 128

    reverse_next = False

    # Calculate for columns
    mod = 1
    for col_bits in format[0]:
        bits = int(col_bits)
        chunk = s[start:start + bits]
        current_representation = _cached_representations_4[i]
        if reverse_next:
            current_representation = reverse_cached_representation(current_representation)
        rank = current_representation.get(chunk, 1)
        mod *= len(current_representation)
        #print("col",col_bits,"modulo ",mod)
        col_result += rank * (n_col // mod)
        reverse_next = (rank % 2 == 0)
        start += bits
        i += 1
    col_result -= 120

    # Calculate for lines
    mod = 1
    for line_bits in format[1]:
        bits = int(line_bits)
        chunk = s[start:start + bits]
        current_representation = _cached_representations_4[i]
        if reverse_next:
            current_representation = reverse_cached_representation(current_representation)
        rank = current_representation.get(chunk, 1)
        mod *= len(current_representation)
        line_result += rank * (n_line // mod)
        reverse_next = (rank % 2 == 0)
        start += bits
        i += 1

    return [col_result, line_result]

def group_720(G, s: str, format=format_5) -> np.array:

    start = 0
    i = 0
    col_result = 0
    line_result = 0
    n_col = 56
    n_line = 720

    reverse_next = False

    # Calculate for columns
    mod = 1
    for col_bits in format[0]:
        bits = int(col_bits)
        chunk = s[start:start + bits]
        current_representation = _cached_representations_5[i]
        if reverse_next:
            current_representation = reverse_cached_representation(current_representation)
        rank = current_representation.get(chunk, 1)
        mod *= len(current_representation)
        col_result += rank * (n_col // mod)
        reverse_next = (rank % 2 == 0)
        start += bits
        i += 1
    col_result -= 8

    # Calculate for lines
    mod = 1
    for line_bits in format[1]:
        bits = int(line_bits)
        chunk = s[start:start + bits]
        current_representation = _cached_representations_5[i]
        if reverse_next:
            current_representation = reverse_cached_representation(current_representation)
        rank = current_representation.get(chunk, 1)
        mod *= len(current_representation)
        line_result += rank * (n_line // mod)
        reverse_next = (rank % 2 == 0)
        start += bits
        i += 1

    line_result -= 51
    return [col_result, line_result]

def test_group_1_neighbors_1bit():
    fmt = format_1
    total_bits = sum(fmt[0]) + sum(fmt[1])
    # Choisir un mot binaire fixé
    s = "".join(np.random.choice(["0", "1"], size=total_bits))
    results = []
    # Stocker aussi le résultat du mot fixé
    fixed_result = group_384_true(None, s, fmt)
    results.append((s, fixed_result))
    for i in range(total_bits):
        # Flip le i-ème bit
        s_list = list(s)
        s_list[i] = "1" if s_list[i] == "0" else "0"
        neighbor = "".join(s_list)
        result = group_384_true(None, neighbor, fmt)
        results.append((neighbor, result))
    print(f"Résultats pour chaque voisin à 1 bit de '{s}' (incluant le mot fixé):")
    for neighbor, result in results:
        print(f"{neighbor} -> {result}")
    # Optionnel: vérifier qu'il y a bien total_bits+1 résultats
    assert len(results) == total_bits + 1, f"On devrait avoir {total_bits + 1} résultats, mais on a {len(results)}"


# Test pour vérifier qu'on a bien 384 chiffres différents pour tous les mots binaires possibles
def test_group_1_unique_outputs():
    fmt = format_3
    # Calculer le nombre total de bits nécessaires
    total_bits = sum(fmt[0]) + sum(fmt[1])
    seen = set()
    for i in range(2**total_bits):
        # Générer le mot binaire de la bonne longueur
        s = f"{i:0{total_bits}b}"
        result = group_384(None, s, fmt)
        # On ne considère que le résultat colonne (col_result)
        seen.add(result[1])
    print(f"Nombre de résultats différents: {len(seen)}")
    print(seen)
    assert len(seen) == 384, f"On devrait avoir 384 résultats différents, mais on a {len(seen)}"

#test_group_1_unique_outputs()
#test_group_1_neighbors_1bit()
