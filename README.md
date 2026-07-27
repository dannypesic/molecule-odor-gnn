# molecule-odor-gnn

A Graph Neural Network that predicts the odor of a molecule from its structure. Input a SMILES string and recieve a probability for each of 138 odor descriptors.

## How do I use it?

```bash
git clone https://github.com/dannypesic/molecule-odor-gnn
cd molecule-odor-gnn
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 inference.py
```

`inference.py` holds a list of molecules of the form `("Name", "SMILES String")` which can be edited for testing.

The pre-trained weights, the dataset, and the formatted data JSON are all committed here for ease of retraining (~5.3 MB total). To retrain: `cd src && python train.py`.

## Can I use any molecule?

You should be able to. Due to the architecture of the model, there is no real limit to molecular dimensions. Large molecules tend to not be odorous, so the model will classify them as odorless.

## Results

| Molecule | SMILES | Prediction |
|---|---|---|
| Geraniol | `CC(C)=CCC/C(C)=C/CO` | floral 0.82 · rose 0.72 · sweet 0.49 · fruity 0.45 · citrus 0.39 |
| Allicin | `C=CCS(=O)SCC=C` | alliaceous 0.91 · onion 0.88 · garlic 0.88 · sulfurous 0.72 · vegetable 0.65 |
| Butyric acid | `CCCC(=O)O` | cheesy 0.85 · sour 0.77 · sweaty 0.37 · sharp 0.27 · dairy 0.25 |
| Limonene | `C=C(C)C1CCC(=CC1)C` | herbal 0.58 · citrus 0.45 · woody 0.42 · pine 0.35 · fresh 0.31 |

## How accurate is it?

```bash
./accuracy.sh                    # Accuracy: 97.7%; threshold 0.5
THRESHOLD=0.3 ./accuracy.sh      # sweep the decision cutoff
```

**97.7% Hamming accuracy** on 10 molecules verified absent from the training set (structural isomorphism check against all 4,983 entries). The script scores each label slot per molecule at a default probability threshold of 0.5.

## How does the math work?

Using the [pysmiles](https://github.com/pckroon/pysmiles) library, a SMILES string is converted into a [Networkx](https://networkx.org/en/) graph where each node contains a feature vector of element number, aromaticity, isotope, hydrogen count, and charge. A Laplacian matrix is created from the graph's adjacency and degree matrices. Then, a Laplacian polynomial of degree 2 is made with its coefficients as parameters, which acts on each feature vector. This creates a series of layers that map the graph to the "convoluted graph," which is then turned into an "adjacency tensor." This tensor is the adjacency matrix of the graph, except each value of 1 is replaced by the two corresponding feature vectors. Finally, a series of convolutions runs over this tensor to create the output vector.

## Dataset

Training data from the Principal Odor Map dataset: 4,983 molecules with 138 odor
descriptors, derived from GoodScents and Leffingwell PMP 2001.

> Lee et al. (2023). A principal odor map unifies diverse tasks in human olfactory
> perception. *Science*, 381(6661), 999–1006. https://doi.org/10.1126/science.ade4401

CSV sourced from [ARY2260/openpom](https://github.com/ARY2260/openpom).
