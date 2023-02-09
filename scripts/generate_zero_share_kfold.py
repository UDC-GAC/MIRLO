#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random

random.seed(42)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_pairs(pairs_filename):
    if not os.path.exists(pairs_filename):
        raise FileNotFoundError(f'Pairs file not found: {pairs_filename}')
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f:
            pair = line.strip().split('\t')
            pairs.append(pair)
    return pairs

def load_fasta(fasta_filename):
    if not os.path.exists(fasta_filename):
        raise FileNotFoundError(f'FASTA file not found: {fasta_filename}')
    seqs = {}
    with open(fasta_filename, 'r') as f:
        for line in f:
            if line.startswith('>'):
                seq_name = line.strip()[1:]
                seqs[seq_name] = ''
            else:
                seqs[seq_name] += line.strip()
    return seqs

def generate_folds(pairs):
    pairs = [pair for pair in pairs if pair[2] == '1']

    rna_interactions = {}
    for pair in pairs:
        if not pair[1] in rna_interactions:
            rna_interactions[pair[1]] = 0
        rna_interactions[pair[1]] += 1

    rna_interactions = sorted(rna_interactions.items(), key=lambda x: x[1], reverse=True)

    folds = [[] for _ in range(5)]
    while len(rna_interactions) > 0:
        k = folds.index(min(folds, key=len))
        rna, _ = rna_interactions.pop()
        remove_proteins = []
        for pair in pairs:
            if pair[1] == rna:
                folds[k % 5].append(pair)
                remove_proteins.append(pair[0])
        pairs = [pair for pair in pairs if pair[0] not in remove_proteins]

    for fold in folds:
        rnas = list(set([pair[1] for pair in fold]))
        proteins = list(set([pair[0] for pair in fold]))
        negative_pairs = []
        for rna in rnas:
            for protein in proteins:
                if not any([pair[0] == protein and pair[1] == rna for pair in fold]):
                    negative_pairs.append([protein, rna, '0'])
        fold.extend(random.sample(negative_pairs, len(fold)))

    ret_folds = [[[], []] for _ in range(5)]
    for k in range(5):
        for i in range(5):
            if i != k:
                ret_folds[k][0].extend(folds[i])
            else:
                ret_folds[k][1].extend(folds[i])
    for fold in ret_folds:
        random.shuffle(fold[0])
        random.shuffle(fold[1])

    return ret_folds

if __name__ == '__main__':
    datasets = ['RPI369', 'RPI488', 'RPI1807', 'RPI2241', 'NPInter']
    for dataset in datasets:
        dataset_path = os.path.join(BASE_PATH, 'data', dataset)
        pairs = load_pairs(os.path.join(dataset_path, 'pairs.txt'))
        rnas = load_fasta(os.path.join(dataset_path, 'rna.fa'))
        proteins = load_fasta(os.path.join(dataset_path, 'pro.fa'))
        rna_structs = load_fasta(os.path.join(dataset_path, 'rna-struct.fa'))
        protein_structs = load_fasta(os.path.join(dataset_path, 'pro-struct.fa'))
        folds = generate_folds(pairs)
        for k, fold in enumerate(folds):
            fold_path = os.path.join(dataset_path, 'kfold_zero_share', f'{k}')
            train_path = os.path.join(fold_path, 'train')
            test_path = os.path.join(fold_path, 'test')
            os.makedirs(train_path, exist_ok=True)
            os.makedirs(test_path, exist_ok=True)
            train_pairs_list, val_pairs_list = fold
            train_pairs = os.path.join(train_path, 'pairs.txt')
            train_rna = os.path.join(train_path, 'rna.fa')
            train_protein = os.path.join(train_path, 'pro.fa')
            train_rna_struct = os.path.join(train_path, 'rna-struct.fa')
            train_protein_struct = os.path.join(train_path, 'pro-struct.fa')
            with \
                open(train_pairs, 'w') as f, \
                open(train_rna, 'w') as f_rna, \
                open(train_protein, 'w') as f_pro, \
                open(train_rna_struct, 'w') as f_rna_struct, \
                open(train_protein_struct, 'w') as f_pro_struct \
            :
                for pair in train_pairs_list:
                    f.write(f'{pair[0]}\t{pair[1]}\t{pair[2]}\n')
                    f_rna.write(f'>{pair[1]}\n{rnas[pair[1]]}\n')
                    f_pro.write(f'>{pair[0]}\n{proteins[pair[0]]}\n')
                    f_rna_struct.write(f'>{pair[1]}\n{rna_structs[pair[1]]}\n')
                    f_pro_struct.write(f'>{pair[0]}\n{protein_structs[pair[0]]}\n')
            val_pairs = os.path.join(test_path, 'pairs.txt')
            val_rna = os.path.join(test_path, 'rna.fa')
            val_protein = os.path.join(test_path, 'pro.fa')
            val_rna_struct = os.path.join(test_path, 'rna-struct.fa')
            val_protein_struct = os.path.join(test_path, 'pro-struct.fa')
            with \
                open(val_pairs, 'w') as f, \
                open(val_rna, 'w') as f_rna, \
                open(val_protein, 'w') as f_pro, \
                open(val_rna_struct, 'w') as f_rna_struct, \
                open(val_protein_struct, 'w') as f_pro_struct \
            :
                for pair in val_pairs_list:
                    f.write(f'{pair[0]}\t{pair[1]}\t{pair[2]}\n')
                    f_rna.write(f'>{pair[1]}\n{rnas[pair[1]]}\n')
                    f_pro.write(f'>{pair[0]}\n{proteins[pair[0]]}\n')
                    f_rna_struct.write(f'>{pair[1]}\n{rna_structs[pair[1]]}\n')
                    f_pro_struct.write(f'>{pair[0]}\n{protein_structs[pair[0]]}\n')
