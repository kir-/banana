# BANANA

Implementation of BANANA (BAsic NeurAl Network for binding Affinity), as described in the [BigBind paper](https://chemrxiv.org/engage/chemrxiv/article-details/6384ad70c5675357f89943c5).

## Dependencies

In addition to pip installing the requirements, you'll need to install torch, dgl, and rdkit.

## Running with pretrained weights

Once you've installed all the dependencies, you're ready to run the pretrained model. First, create a PDB file containing the pocket residues of your target protein. Then create a smi file with the SMILES strings of the compounds you want to screen (one smiles string per line). Now simply run `python inference.py compounds.smi pocket.pdb --out_file out_file.txt`. This will load the pretrained weights from `data/banana_final.pt`, score each compound in `compounds.smi`, and write all the scores to `out_file.txt`.

If the inference script fails to load any smiles in the dataset, it will output -100 for the BANANA score.

Colab [here](https://colab.research.google.com/drive/1XyLrBIRzMlpdPZUHV2vCFDEZz2xuqIi0?usp=sharing). You can use this to run inference on crossdocked2020.

## Training

If you want to train the model yourself, first make sure you've downloaded the [BigBind dataset](https://storage.googleapis.com/bigbind/BigBindV1.tar.bz2).

Now create a file `configs/local.yaml`. This contains all the configuration that should differ per-computer. Add in this information:
```yaml

project: "Your wandb project (optional)"

platform:
  bigbind_dir: "/path/to/bigbind/folder"
  # lit_pcba_dir is optional, only needed if you want to test your model on LIT_PCBA
  lit_pcba_dir: "/path/to/lit_pcba/folder"
  # To speed up dataloading and other random things, many pickles are saved.
  # cache_dir specifies where they should be saved to
  cache_dir: "/path/to/cache/dir"
  num_workers: num_workers_for_your_dataloader

```

Now that you have this, train a model py running `python train.py config_name overrides`. The config name used in the BigBind paper is `classification`. This config is found in `configs/classification.yaml`. Feel free to make your own config. Anything in the config file can be overwritten with command line arguments. For instance, train with a batch size of 2 with `python train.py classification batch_size=2`.

If you have a Weights and Biases project and specify it in the local config file, the training script will log all the weights and metrics.

## Validation

The validation scripts are all designed to download models from Weights and Biases. Unfortunately, you'll need to modify them if you don't want to use wandb.

To validate the model on any of the BigBind data splits, run `python -m validation.validate run_id=wandb_run_id tag=tag_of_weights_artifact data_split=train|val|test`.

If you want to benchmark the model on the BigBind screening benchmarks, run `python -m validation.screen run_id=wandb_run_id tag=tag_of_weights_artifact benchmark=bigbind data_split=val|test`. To benchmark on LIT-PCBA, you'll need to first run `preprocess_lit_pcba.py` to produce pocket files for all the targets (note: this script requires PyMol). Once you've done this, run `python -m validation.screen run_id=wandb_run_id tag=tag_of_weights_artifact benchmark=lit_pcba`. Both these scripts will produce csv files in `outputs/`.

If you have any questions, email me at [mixarcid@unc.edu](mailto:mixarcid@unc.edu).


