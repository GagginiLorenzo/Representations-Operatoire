"""
Base label-space representation for custom blocks-graycode.
One could implement simple graycode and use straightforward decoding,
but this experimental version was used initially and remains for legacy reasons.
A better and more efficient approach will be implemented in the future.
"""

# ranking_TSP_1/group_representations/C3_representation.py
import numpy as np

def apply_permutation(point, permutation):
    """Applique une permutation à un point."""
    return tuple(map(int, np.dot(permutation, point)))

def generate_permutations_list(initial_point, matrix_sequence):
    """Génère une liste de permutations à partir d'une séquence de matrices."""
    permutations = []
    current_point = np.array(initial_point)

    for permutation_matrix in matrix_sequence:
        current_point = apply_permutation(current_point, permutation_matrix)
        if tuple(current_point) not in permutations:
            permutations.append(tuple(current_point))
        else:
            print("Duplicate found:", tuple(current_point), 'at', len(permutations))
            break
    return permutations

def swap_neg1_to_0(permutations):
    """Remplace -1 par 0 dans une liste de permutations."""
    return [tuple(0 if x == -1 else x for x in perm) for perm in permutations]


def generate_cube_matrix(n, i):
    P = np.eye(n, dtype=int)
    if i == 0:
        P[[i, i]] = -1*P[[i, i]]
        return P
    else:
        P[[i, i]] = -1*P[[i, i]]
        P[[i-1, i]] = P[[i, i-1]]
    return P

def generate_cube_matrices(n):
    matrices = {}
    for i in range(0, n):          # Start index (1-based)
        matrices[i] = generate_cube_matrix(n, i)
    return matrices

def pattern_dict(size):
    pattern_dict = {
        "1": [['0', '0']],
        "2": [['0', '1', '0', '1']],
        "3": [['2', '0', '2', '0', '2', '0', '2', '0']],
        "4": [['3', '0', '3', '0', '3', '1', '3', '2', '3', '1', '3', '0', '3', '0', '3', '1']],
        "5": [['4', '0', '4', '1', '4', '2', '4', '1']*4],
        "6": [['5', '0', '5', '0', '5', '3', '5', '3', '5', '0', '5', '0', '5', '1', '5', '3']*4],
        "7": [['6', '0', '6', '0', '6', '2', '6', '2', '6', '0', '6', '1', '6', '4', '6', '1',
               '6', '1', '6', '0', '6', '4', '6', '1', '6', '0', '6', '0', '6', '1', '6', '4']*4]
    }
    return pattern_dict.get(str(size), [])

def tuple_to_string(tup):
    """Convert a tuple of integers to a string."""
    return ''.join(str(x) for x in tup)

def Cn_representation(size, truncated=False):
    """Representation of the group Cn."""
    initial_point = np.array([-1] * size)

    n = size
    matrices = generate_cube_matrices(n)

    # Fetch the pattern string for the given size
    pattern_string = pattern_dict(size)
    if not pattern_string:
        raise ValueError(f"No pattern defined for size {size}")

    # Convert the pattern string to a list of integers
    pattern = [int(i) for i in pattern_string[0]]

    # Generate the pattern matrices
    pattern_matrices = [matrices[i] for i in pattern]

    # Generate permutations
    permutations_list = generate_permutations_list(initial_point, pattern_matrices)
    permutations_list_swapped = swap_neg1_to_0(permutations_list)

    if truncated:
        # Remove the last element from the list
        permutations_list_swapped = permutations_list_swapped[:-1]

    # Convert the output to a dictionary format (tuple: rank, starting from 1)
    output_dict = {tuple_to_string(tup): (rank + 2 if rank < len(permutations_list_swapped) - 1 else 1) for rank, tup in enumerate(permutations_list_swapped)}

    # Convert tuples to strings
    output_dict_strings = {tuple_to_string(tup): rank for tup, rank in output_dict.items()}

    return output_dict_strings

## Example usage
#output = Cn_representation(1, truncated=False)

# Print the output
#for permutation, rank in output.items():
#    print(f"Permutation {permutation}: Rank {rank}")
