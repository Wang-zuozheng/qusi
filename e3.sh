export CUDA_VISIBLE_DEVICES='1'
python main.py -datasets 'coco' -is_ft 1 -is_af 0 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0.1 -st 60 -wr 0 -T 0.3
python main.py -datasets 'coco' -is_ft 1 -is_af 0 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0.3 -st 60 -wr 0 -T 0.3
python main.py -datasets 'coco' -is_ft 1 -is_af 0 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0.4 -st 60 -wr 0 -T 0.3
python main.py -datasets 'coco' -is_ft 1 -is_af 0 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0.5 -st 60 -wr 0 -T 0.3
python main.py -datasets 'coco' -is_ft 1 -is_af 0 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0.6 -st 60 -wr 0 -T 0.3
python main.py -datasets 'coco' -is_ft 1 -is_af 0 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0.7 -st 60 -wr 0 -T 0.3
python main.py -datasets 'coco' -is_ft 1 -is_af 0 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0.8 -st 60 -wr 0 -T 0.3
python main.py -datasets 'coco' -is_ft 1 -is_af 0 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0.9 -st 60 -wr 0 -T 0.3