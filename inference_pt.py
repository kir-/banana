#!/usr/bin/env python

import argparse
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm

from terrace.comp_node import Input
from terrace.batch import make_batch_td, DataLoader

from datasets.inference_dataset import InferenceDataset
from datasets.data_types import IsActiveData
from models.make_model import make_model
from rdkit import Chem

def inference():

    parser = argparse.ArgumentParser(description="BANANA inference")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers in the dataloader")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--no_gpu", dest='no_gpu', action='store_true', help="Disable GPU (only use CPU)")
    parser.add_argument("--out_file", type=str, default="out.csv", help="File to store the output scores")
    parser.set_defaults(no_gpu=False)
    args = parser.parse_args()

    # device = "cpu" if args.no_gpu else "cuda:0"
    device = "cpu"
    data = torch.load('split_by_name.pt')
    data_path = 'data/crossdocked_pocket10'
    train_data = data['train']
    test_data = data['test']
    combined_data = train_data + test_data
    
    cfg = OmegaConf.load("configs/classification.yaml")
    in_node = Input(make_batch_td(IsActiveData.get_type_data(cfg)))
    
    model = make_model(cfg, in_node)
    model.load_state_dict(torch.load("data/banana_final.pt"))
    model = model.to(device)
    model.eval()

    with open(args.out_file, "a") as f:
        # TODO Generalize dataset to pocket or multiple pockets
        for entry in tqdm(combined_data, desc="Inference on protein + ligand", unit="ligand"):
            pocket_name, ligand_name = entry
            pocket = os.path.join(data_path, pocket_name)
            ligand = os.path.join(data_path, ligand_name)
            mol = Chem.SDMolSupplier(ligand)[0]
            smi = Chem.MolToSmiles(mol)
            dataset = InferenceDataset(cfg, smi, pocket)
            dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                    num_workers=args.num_workers, pin_memory=True,
                                    shuffle=False)

            ligand_txt = ligand_name.split('/').replace('/','-')[:-3] + 'txt'
            for batch in tqdm(dataloader):
                batch = batch.to(device)
                output = model(batch).cpu().numpy()
                for x, out in zip(batch, output):
                    if not x.is_active:
                        out = -100
                    f.write(f"{ligand_txt},{out}\n")

if __name__ == "__main__":
    with torch.no_grad():
        inference()

