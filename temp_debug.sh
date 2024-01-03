export CUDA_VISIBLE_DEVICES=1
python main.py -datasets 'coco' -is_ft 1 -is_af 0 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0.2 -st 60 -wr 0 -T 0.1
python main.py -datasets 'coco' -is_ft 1 -is_af 0 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0.2 -st 60 -wr 0 -T 0.2
python main.py -datasets 'coco' -is_ft 1 -is_af 0 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0.2 -st 60 -wr 0 -T 0.3
python main.py -datasets 'coco' -is_ft 1 -is_af 0 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0.2 -st 60 -wr 0 -T 0.4
python main.py -datasets 'coco' -is_ft 1 -is_af 0 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0.2 -st 60 -wr 0 -T 0.5
python main.py -datasets 'coco' -is_ft 1 -is_af 0 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0.2 -st 60 -wr 0 -T 0.7
