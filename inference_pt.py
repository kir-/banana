#!/usr/bin/env python

import argparse
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
import logging
from torch.profiler import profile, ProfilerActivity


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

    device = "cpu" if args.no_gpu else "cuda:0"
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
    profiler_log_file = "profiler_results.txt"
    with open(profiler_log_file, "w") as profiler_log:
        with open(args.out_file, "a") as f:
            # Warning this runs on per tuple of a pt file and therefore is very slow
            # TODO Generalize dataset to run on pocket or multiple pockets
            for i, entry in enumerate(tqdm(combined_data, desc="Inference on protein + ligand", unit="ligand")):
                pocket_name, ligand_name = entry
                pocket = os.path.join(data_path, pocket_name)
                ligand = os.path.join(data_path, ligand_name)
                mol = Chem.SDMolSupplier(ligand)[0]
                smi = Chem.MolToSmiles(mol)
                if i % 1000 == 0:
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, use_cuda=not args.no_gpu) as prof:
                        pocket_name, ligand_name = entry
                        try:
                            dataset = InferenceDataset(cfg, smi, pocket)
                        except:
                            logging.warning("{pocket_name}: pdb incorrect format")
                        dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                                num_workers=args.num_workers, pin_memory=True,
                                                shuffle=False)

                        ligand_txt = ligand_name.replace('/','-')[:-3] + 'txt'
                        
                        for batch in tqdm(dataloader):
                            batch = batch.to(device)
                            output = model(batch)
                                        
                            # output = model(batch).cpu().numpy()
                            for x, logits in zip(batch, output):
                                prob = torch.sigmoid(logits).cpu().numpy()
                                if not x.is_active:
                                    out = -100
                                f.write(f"{ligand_txt},{prob}\n")
                    profiler_log.write(f"Profiling results for {entry}:\n")
                    profiler_log.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
                    profiler_log.write("\n\n")  # Add some space between entries
                else:
                    ligand_txt = ligand_name.replace('/','-')[:-3] + 'txt'
                        
                    for batch in tqdm(dataloader):
                        batch = batch.to(device)
                        output = model(batch)
                                        
                        # output = model(batch).cpu().numpy()
                        for x, logits in zip(batch, output):
                            prob = torch.sigmoid(logits).cpu().numpy()
                            if not x.is_active:
                                out = -100
                            f.write(f"{ligand_txt},{prob}\n")

if __name__ == "__main__":
    with torch.no_grad():
        inference()

