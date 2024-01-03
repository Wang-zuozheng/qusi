export CUDA_VISIBLE_DEVICES='2'
python main.py -datasets 'coco' -is_ft 0 -is_af 1 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0.2 -st 80 -wr 0 -T 0.3
