"""
Training-time diagnostic metrics for sequence VAEs.

`token_reconstruction_accuracy` is called by `SelfiesVAE.fine_tune`
every batch to report per-epoch decoder accuracy.
"""

from rdkit import Chem


def validity(smiles_list):
    valid = 0
    for smi in smiles_list:
        if Chem.MolFromSmiles(smi) is not None:
            valid += 1
    return valid / len(smiles_list) if smiles_list else 0.0


def uniqueness(smiles_list):
    if not smiles_list:
        return 0.0
    return len(set(smiles_list)) / len(smiles_list)


def novelty(gen_smiles, train_smiles):
    train_set = set(train_smiles)
    novel = [s for s in gen_smiles if s not in train_set]
    return len(novel) / len(gen_smiles) if gen_smiles else 0.0


def token_reconstruction_accuracy(pred, target, pad_token_id=-100):
    mask = (target != pad_token_id)
    if mask.sum() == 0:
        return 0.0
    correct = (pred == target) & mask
    return correct.sum().item() / mask.sum().item()
