import argparse
import numpy as np
import math
import timeit
from typing import Dict, List, Tuple
import newick
from scipy.linalg import expm

class Node:
    def __init__(self, name, left=None, right=None, branch_length=0.0, branch_id=0):
        self.name = name
        self.left = left
        self.right = right
        self.branch_length = branch_length
        self.branch_id = branch_id
        self.probs = None

def parse_tree(filename: str) -> Tuple[Node, List[Node]]:
    """Parse a Newick format tree file and return the root node and a post-order traversal list.
    
    Arguments:
        filename: Path to the Newick format tree file
    Returns:
        root: Root node of the tree
        ordering: List of nodes in post-order traversal
    """
    with open(filename) as f:
        tree = newick.load(f)[0]
    
    def create_node(node, branch_id=0):
        if node.is_leaf:
            return Node(node.name, None, None, node.length or 0.0, branch_id)
        
        left = create_node(node.descendants[0], branch_id + 1)
        right = create_node(node.descendants[1], branch_id + 2)
        return Node(node.name, left, right, node.length or 0.0, branch_id)
    
    root = create_node(tree)
    
    ordering = []
    def post_order(node):
        if node.left:
            post_order(node.left)
        if node.right:
            post_order(node.right)
        ordering.append(node)
    
    post_order(root)
    return root, ordering

def read_data(filename: str) -> Tuple[Dict[str, str], int]:
    '''Reads data from filename in fasta format.

    Arguments:
        filename: name of fasta file to read
    Returns:
        sequences: dictionary of outputs (string (sequence id) -> sequence (string))
        size: length of each sequence
    '''
    sequences = {}
    curr_id = None
    curr_seq = []
    
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if curr_id is not None:
                    sequences[curr_id] = ''.join(curr_seq)
                curr_id = line[1:].strip()
                curr_seq = []
            else:
                curr_seq.append(line)
                
    if curr_id is not None:
        sequences[curr_id] = ''.join(curr_seq)
    
    size = len(next(iter(sequences.values()))) if sequences else 0
    return sequences, size

def create_rate_matrix_HKY(pi, kappa):
    """Create HKY rate substition matrix."""
    bases = 'ACGT'
    Q = np.zeros((len(bases), len(bases)))
    purines = [bases.index('A'), bases.index('G')]
    pyrimidines = [bases.index('C'), bases.index('T')]
    
    for i in range(4):
        for j in range(4):
            if i != j:
                if (i in purines and j in purines) or (i in pyrimidines and j in pyrimidines):
                    Q[i, j] = kappa * pi[j]
                else:
                    Q[i, j] = pi[j]
    for i in range(4):
        Q[i, i] = -np.sum(Q[i, :])
    
    return Q


def pruning(Q, pi, root, site_data, scale_factor):
    """Felsenstein's pruning algorithm."""

    nucleotide_map = {
        "A": 0,
        "C": 1,
        "T": 2,
        "G": 3
    }
    
    def post_order_traverse(node):
        if node.left is None and node.right is None:
            if node.name in site_data:
                nuc = site_data[node.name]
                if nuc in nucleotide_map and nucleotide_map[nuc] is not None:
                    probs = np.zeros(4)
                    probs[nucleotide_map[nuc]] = 1.0
                else:
                    probs = np.ones(4) / 4
                return probs
            return np.ones(4) / 4
            
        left_probs = post_order_traverse(node.left)
        right_probs = post_order_traverse(node.right)
        
        scaled_left_length = node.left.branch_length * scale_factor
        scaled_right_length = node.right.branch_length * scale_factor
        
        P_left = expm(Q * scaled_left_length)
        P_right = expm(Q * scaled_right_length)
        
        return (P_left @ left_probs) * (P_right @ right_probs)

    root_probs = post_order_traverse(root)
    likelihood = np.sum(pi * root_probs)
    return np.log(likelihood) if likelihood > 0 else float('-inf')

''' Outputs the Viterbi decoding of a given observation.
Arguments:
	obs: observed sequence of emitted states (list of emissions)
	trans_probs: transition log-probabilities (dictionary of dictionaries)
	emiss_probs: emission log-probabilities (dictionary of dictionaries)
	init_probs: initial log-probabilities for each hidden state (dictionary)
Returns:
	l: list of most likely hidden states at each position
        (list of hidden states)
	p: log-probability of the returned hidden state sequence
'''
def viterbi(obs, trans_probs, emiss_probs, init_probs):
    ''' Complete this function. '''

    # Adapted from Assignment 2
    obs_length = len(obs)
    m = {
        'h': [None for i in range(obs_length)],
        'l': [None for i in range(obs_length)]
    }
    states_path = {
        'h': [None for i in range(obs_length)],
        'l': [None for i in range(obs_length)]
    }
    m['h'][0] = np.log(init_probs['h']) + emiss_probs[0, 0]  
    m['l'][0] = np.log(init_probs['l']) + emiss_probs[1, 0]  
    for t in range(1, obs_length):
        if (m['h'][t-1] + np.log(trans_probs['h']['h'])) > (m['l'][t-1] + np.log(trans_probs['l']['h'])):
            m['h'][t] = (m['h'][t-1] + np.log(trans_probs['h']['h'])) + emiss_probs[0, t]
            states_path['h'][t] = 'h'
        else:
            m['h'][t] = (m['l'][t-1] + np.log(trans_probs['l']['h'])) + emiss_probs[0, t]
            states_path['h'][t] = 'l'
        if (m['h'][t-1] + np.log(trans_probs['h']['l'])) > (m['l'][t-1] + np.log(trans_probs['l']['l'])):
            m['l'][t] = (m['h'][t-1] + np.log(trans_probs['h']['l'])) + emiss_probs[1, t]
            states_path['l'][t] = 'h'
        else:
            m['l'][t] = (m['l'][t-1] + np.log(trans_probs['l']['l'])) + emiss_probs[1, t]
            states_path['l'][t] = 'l'
    final_h = m['h'][-1]
    final_l = m['l'][-1]
    if final_h > final_l:
        p = final_h
        final_state = 'h'
    else:
        p = final_l
        final_state = 'l'
        
    l = [None] * obs_length
    l[-1] = final_state
    for t in range(obs_length - 1, 0, -1):
        l[t-1] = states_path[l[t]][t]
    return l, p

def find_intervals(sequence: List[str]) -> List[Tuple[int, int]]:
    """Find intervals of high GC content states."""
    intervals = []
    start = None
    
    for i, state in enumerate(sequence, 1):
        if state == 'h' and start is None:
            start = i
        elif state == 'l' and start is not None:
            intervals.append((start, i-1))
            start = None
    
    if start is not None:
        intervals.append((start, len(sequence)))
    
    return intervals

def main():
    """Run PhyloHMM algorithm."""
    parser = argparse.ArgumentParser(description='PhyloHMM.')
    parser.add_argument('-f', type=str, required=True, help='Input FASTA file')
    parser.add_argument('-out', type=str, required=True, help='Output')
    parser.add_argument('-tree', type=str, required=True, help='Tree file')
    args = parser.parse_args()

    with open(args.out, "w") as f:
        f.write("\n")
    sequences, length = read_data(args.f)
    root, ordering = parse_tree(args.tree)
    species = ['Canis_lupus_familiaris', 'Canis_latrans', 'Vulpes_lagopus', 'Felis_catus', 'Felis_chaus', 'Mus_musculus', 'Rattus_norvegicus', 'Ursus_arctos', 'Ursus_maritimus', 'Homo_sapiens']


    transition_probabilities = {
        'h': {'h': 0.9, 'l': 0.1},
        'l': {'h': 0.05, 'l': 0.95}
    }
    
    #https://www.math.clemson.edu/~macaule/classes/f16_math4500/slides/f16_math4500_cpg-islands_handout.pdf 
    pi_h = np.array([0.15, 0.33, 0.36, 0.16])
    pi_l = np.array([0.27, 0.24, 0.26, 0.23])

    #kappa = transition/transversion ratio, normally 2.0 in humans
    #https://rosalind.info/glossary/transitiontransversion-ratio/#:~:text=The%20transition%2Ftransversion%20ratio%20between,mutation%20in%20the%20translated%20protein. 
    kappa = 2.0

    Q_h = create_rate_matrix_HKY(pi_h, kappa)
    Q_l = create_rate_matrix_HKY(pi_l, kappa)

    for seq_id, seq in sequences.items():
        print('\n')
        print(seq_id)
        emission_likelihoods = np.zeros((2, length))
        for pos in range(length):
            site_data = {seq_id: seq[pos]}
            emission_likelihoods[0, pos] = pruning(Q_h, pi_h, root, site_data, scale_factor = 1.0)
            emission_likelihoods[1, pos] = pruning(Q_l, pi_l, root, site_data, scale_factor = 5.0)
        initial_probabilities = {'h': 0.5, 'l': 0.5}
        state_path, log_prob = viterbi(seq, transition_probabilities, emission_likelihoods, initial_probabilities)
    
        cpg_intervals = find_intervals(state_path)

        with open(args.out, "a") as f:
            reference_seq = next(iter(sequences.keys()))
            f.write(f">{seq_id}\n")
            f.write("\n".join([f"{start},{end}" for (start, end) in cpg_intervals]))
            f.write("\n")
        
        print(f"\nFound {len(cpg_intervals)} CpG islands")
        print(f"Log probability of optimal path: {log_prob:.2f}")
        print("CpG island locations:")
        for start, end in cpg_intervals:
            print(f"{start}-{end}")

if __name__ == "__main__":
    start = timeit.default_timer()
    main()
    end = timeit.default_timer()
    print(f"\nTotal time: {end - start:.2f} seconds")