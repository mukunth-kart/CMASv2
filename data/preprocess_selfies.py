"""
SELFIES preprocessing entry point (canonical filename).

Delegates to :mod:`data.smiles_to_selfies` which hosts the actual
SMILES->SELFIES conversion, vocab building, and `.npy` tokenization
logic.  Both filenames exist so the repo remains compatible with either
naming convention without duplicating code.

Usage is identical to `smiles_to_selfies.py`:

    python data/preprocess_selfies.py \
        --input ./data/ChemBL_Smiles.txt \
        --out_npy ./data/ChemBL_Selfies.npy \
        --vocab_out ./vocab/selfies_vocab.json --build_vocab
"""

from smiles_to_selfies import preprocess, parse_args


if __name__ == "__main__":
    args = parse_args()
    preprocess(
        input_path=args.input,
        out_npy=args.out_npy,
        vocab_path=args.vocab_out,
        build_vocab=args.build_vocab,
        max_len=args.max_len,
    )
