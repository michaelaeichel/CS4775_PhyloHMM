# CS4775_final_project

### Michaela Eichel (mae97)

PhyloHMM implementation to search for CpG islands in simulated nucleotide sequences of mammals.

To run this project, run in terminal:

python phylo_hmm.py -f synthetic_sequences.fasta -tree tree.nwk -out phylo_hmm_results.txt

Files in this repo:

phylo_hmm.py:

Implementation of Phylogenetic Hidden Markov Models algorithm for finding CpG islands. 
Main function takes in 3 arguments:
        - f: input fasta file of species names and nucleotide sequences
        - tree: input newick tree to relate species in f 
        - out: output txt file of species and any detected CpG islands

PhyloHMM_SeqGen folder:

Simulation of nucleotide sequences using HKY evolution model and Seqgen python package ([text](http://tree.bio.ed.ac.uk/software/seqgen/))

    Sequence_Generation.ipynb: Python notebook of nucleotide simulation process
    phylohmm_data_generation_tree.nwk: Newick tree relating species for input into SeqGen

synthetic_sequences.fasta:

Fasta file of 10000 base pair long simulated nucleotide sequences for 10 mammals for use in phylo_hmm.py

tree.nwk:

Newick tree relating 10 mammals for use in phylo_hmm.py

phylo_hmm_results.txt

Outputted results of running main() in phylo_hmm.py