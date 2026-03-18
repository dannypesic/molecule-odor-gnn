# molecule-odor-gnn
A Graph Neural Network that determines the odor of a molecular substance based on its structure

## How do I use it?
The model `best_model.pth` is pre-trained from `src/train.py`. Try it out by cloning the repository, entering the project directory, and running `python3 inference.py` to see the examples. `inference.py` contains a list of molecules of the form `("Name", "SMILES String")`, which can be manually changed.

Note: both the pre-trained model weights, the dataset, and the formatted data json are included in this repo for ease of retraining. Their collective size is approximately 5.3MB.

## Can I use any molecule?
You should be able to. Due to the architecture of the model, there is no real limit to molecular dimensions. Large molecules tend to not be odorous, so the model will classify them as odorless.

## Does it work?
Yes! It works pretty well! Here are some examples:

**Geraniol** — `CC(C)=CCC/C(C)=C/CO`
- floral: 0.819
- rose: 0.717
- sweet: 0.485
- fruity: 0.454
- citrus: 0.394
- green: 0.313
- waxy: 0.294
- lemon: 0.277
- fresh: 0.269
- lily: 0.265
- herbal: 0.165
- muguet: 0.159
- woody: 0.155

**Limonene** — `C=C(C)C1CCC(=CC1)C`
- herbal: 0.582
- citrus: 0.446
- woody: 0.420
- pine: 0.353
- sweet: 0.327
- fresh: 0.313
- mint: 0.287
- balsamic: 0.283
- floral: 0.276
- terpenic: 0.273
- spicy: 0.269
- green: 0.207
- fruity: 0.202
- lemon: 0.199
- cooling: 0.160

**Allicin** — `C=CCS(=O)SCC=C`
- alliaceous: 0.914
- onion: 0.884
- garlic: 0.876
- sulfurous: 0.718
- vegetable: 0.654
- green: 0.537
- meaty: 0.312
- savory: 0.288
- pungent: 0.226
- roasted: 0.198
- cooked: 0.197
- radish: 0.171

**Hedione** — `COC(=O)CC1CCC(=O)C1CCCC`
- fruity: 0.662
- sweet: 0.315
- floral: 0.303
- green: 0.298
- herbal: 0.279
- jasmin: 0.198
- tropical: 0.183
- woody: 0.179

**Butyric acid** — `CCCC(=O)O`
- cheesy: 0.854
- sour: 0.766
- sweaty: 0.372
- sharp: 0.273
- dairy: 0.248
- fruity: 0.228
- buttery: 0.227

## How does the math work?
Using the [pysmiles](https://github.com/pckroon/pysmiles) library, a SMILES string is converted into a [Networkx](https://networkx.org/en/) graph where each node contains a feature vector of element number, aromaticity, isotope, hydrogen count, and charge. A Laplacian matrix is created from the graph's adjacency and degree matrices. Then, a Laplacian polynomial of degree 2 is made with its coefficients as parameters, which acts on each feature vector. This creates a series of layers that map the graph to the "convoluted graph," which is then turned into an "adjacency tensor." This tensor is the adjacency matrix of the graph, except each value of 1 is replaced by the two corresponding feature vectors. Finally, a series of convolutions runs over this tensor to create the output vector. 

## Improvements?
While accuracy is relatively high, molecular-odor-gnn could be improved by increasing the number of training epochs and using a broader dataset. 

## Dataset

Training data sourced from the [Principal Odor Map dataset](https://github.com/ARY2260/openpom) — 
4,983 molecules with 138 odor descriptors, derived from GoodScents and Leffingwell PMP 2001.

> Lee et al. (2023). A principal odor map unifies diverse tasks in human olfactory perception. 
> *Science*, 381(6661), 999–1006. https://doi.org/10.1126/science.ade4401

## License
Code is MIT licensed. Note that model weights are trained on the 
[OpenPOM dataset](https://github.com/ARY2260/openpom). Consult their 
license before any commercial use.
