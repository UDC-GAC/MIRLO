#!/usr/bin/env bash
# Test MIRLO against the NPInter dataset

SCRIPT_DIR=$(dirname -- "$( readlink -f -- "$0"; )";)
SCRIPT_DIR=$(dirname -- "$( readlink -f -- "$SCRIPT_DIR"; )";)

python3 $SCRIPT_DIR/main.py train \
                --output $SCRIPT_DIR/MIRLO \
                $SCRIPT_DIR/data/NPInter/kfold_no_share_protein/0/train/rna.fa \
                $SCRIPT_DIR/data/NPInter/kfold_no_share_protein/0/train/pro.fa \
                $SCRIPT_DIR/data/NPInter/kfold_no_share_protein/0/train/pairs.txt
python3 $SCRIPT_DIR/main.py evaluate \
                $SCRIPT_DIR/data/NPInter/kfold_no_share_protein/0/test/rna.fa \
                $SCRIPT_DIR/data/NPInter/kfold_no_share_protein/0/test/pro.fa \
                $SCRIPT_DIR/data/NPInter/kfold_no_share_protein/0/test/pairs.txt \
                $SCRIPT_DIR/MIRLO
# python3 $SCRIPT_DIR/main.py predict \
#                 --output $SCRIPT_DIR/MIRLO_predictions.txt \
#                 $SCRIPT_DIR/data/NPInter/kfold_no_share_protein/0/test/rna.fa \
#                 $SCRIPT_DIR/data/NPInter/kfold_no_share_protein/0/test/pro.fa \
#                 $SCRIPT_DIR/MIRLO
rm -rf $SCRIPT_DIR/MIRLO.h5 \
       $SCRIPT_DIR/MIRLO.joblib \
       $SCRIPT_DIR/MIRLO_predictions.txt
