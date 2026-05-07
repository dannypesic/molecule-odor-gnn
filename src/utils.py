from pathlib import Path
import torch
import pysmiles

ELEMENT_TO_NUM = {
    'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10,
    'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,'Cl':17,'Ar':18,
    'K':19,'Ca':20,'Sc':21,'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,
    'Co':27,'Ni':28,'Cu':29,'Zn':30,'Ga':31,'Ge':32,'As':33,'Se':34,
    'Br':35,'Kr':36,'Rb':37,'Sr':38,'Y':39,'Zr':40,'Nb':41,'Mo':42,
    'Tc':43,'Ru':44,'Rh':45,'Pd':46,'Ag':47,'Cd':48,'In':49,'Sn':50,
    'Sb':51,'Te':52,'I':53,'Xe':54,'Cs':55,'Ba':56,
}

# Number of node features; used by model to set INPUT_DIM
NODE_FEATURE_DIM = 5  # element_num, aromatic, isotope, hcount, charge


def smiles_to_graph(smiles: str):
    # Strip E/Z stereo tokens as pysmiles requires both sides to be specified and the GNN doesn't use geometric isomer info anyway
    # This should be fixed ^^! Geometry is very important why did i make it like this
    smiles = smiles.replace("/", "").replace("\\", "")
    return pysmiles.read_smiles(smiles, explicit_hydrogen=False)


def graph_to_tensors(mol):
    nodes   = list(mol.nodes())
    N       = len(nodes)
    node_idx = {n: i for i, n in enumerate(nodes)}

    features = []
    for node in nodes:
        attrs = mol.nodes[node]
        elem_num = float(ELEMENT_TO_NUM.get(attrs.get('element', 'C'), 6))
        aromatic = float(bool(attrs.get('aromatic', False)))
        isotope  = float(attrs.get('isotope') or 0.0)
        hcount   = float(attrs.get('hcount', 0) or 0.0)
        charge   = float(attrs.get('charge', 0) or 0.0)
        features.append([elem_num, aromatic, isotope, hcount, charge])

    x = torch.tensor(features, dtype=torch.float32)  # [N, 5]

    A = torch.zeros(N, N, dtype=torch.float32)
    for u, v in mol.edges():
        i, j = node_idx[u], node_idx[v]
        A[i, j] = 1.0
        A[j, i] = 1.0

    D = torch.diag(A.sum(dim=1))
    L = D - A 

    return x, A, L


def smiles_to_tensors(smiles: str):
    mol = smiles_to_graph(smiles)
    return graph_to_tensors(mol)