import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from functools import lru_cache
from fuzzywuzzy import process


def smiles_to_fingerprint(smiles, radius=5, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    else:
        return np.zeros(n_bits)


with open('gen2vec.txt') as io:
    gen2vec = io.readlines()[1:]
    
gen2vec_mapper = dict()
for gene_info in gen2vec:
    gene, *embed = gene_info.split()
    embed = list(map(float, embed))
    gene = gene.strip()
    gen2vec_mapper[gene] = embed
    
@lru_cache(512)
def get_closest_name_gene(name, threshold=50):
    names = list(gen2vec_mapper.keys())
    closest_matches = process.extractBests(name, names, score_cutoff=threshold)
    if closest_matches:
        return closest_matches[0][0]
    return None


def get_gene_vec(gene):
    gene = gene.strip()
    if gene in gen2vec_mapper:
        return gen2vec_mapper[gene]
    else:
        gene = get_closest_name_gene(gene)
        if gene in gen2vec_mapper:
            return gen2vec_mapper[gene]
        else:
            return [0] * 200