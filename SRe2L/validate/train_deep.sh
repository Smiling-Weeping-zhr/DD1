# wandb disabled
wandb enabled
wandb online

python train_deep.py \
    --wandb-project 'val_rn18_fkd' \
    --batch-size 1024 \
    --gradient-accumulation-steps 4 \
    --model resnet18 \
    --device 'cuda:1' \
    --cos \
    -j 4 \
    -T 20 \
    --mix-type 'cutmix' \
    --output-dir '/home/test_yanjunchi/wangshaobo/DD/SRe2L/SRe2L/validate/save/val_rn18_fkd/rn18_[4K]_T20/' \
    --train-dir '/home/test_yanjunchi/wangshaobo/DD/SRe2L/SRe2L/recover/syn_data/rn18_Imagnet1K/sre2l_in1k_rn18_4k_ipc50/' \
    --val-dir '/data/ILSVRC2012/val' \
    --fkd-path '/home/test_yanjunchi/wangshaobo/DD/SRe2L/SRe2L/relabel/labels/sre2l_in1k_rn18_4k_ipc50/'