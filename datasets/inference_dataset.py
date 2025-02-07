
import torch
from torch.utils import data
from rdkit import Chem

from datasets.graphs.mol_graph import MolGraph
from datasets.graphs.prot_graph import ProtGraph, prot_graph_from_pdb
from datasets.data_types import IsActiveData

class InferenceDataset(data.Dataset):
    """ Dataset for inference script. Just loads a smi file and a pdb file,
    and rins with it """

    def __init__(self, cfg, smi, pdb_file):
        super().__init__()
        self.cfg = cfg
        self.smiles = [smi]
        self.prot_graph = prot_graph_from_pdb(cfg, pdb_file)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, index):
        if index >= len(self):
            raise StopIteration

        is_valid = True
        mol = Chem.MolFromSmiles(self.smiles[index])
        if mol is None:
            print(f"Failed to load smiles '{self.smiles[index]}', skipping...")
            mol = Chem.MolFromSmiles("CCCC")
            is_valid = False

        lig_graph = MolGraph(self.cfg, mol, use_3d=False)
        # hacky -- use the activity column to specify molecule validity
        is_active = torch.tensor(is_valid, dtype=bool)

        return IsActiveData(lig_graph, self.prot_graph, is_active)