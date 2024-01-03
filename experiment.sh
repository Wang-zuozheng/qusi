export CUDA_VISIBLE_DEVICES='4'
# python main.py -datasets 'coco' -is_ft 1 -is_af 0 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0.1 -st 80 -wr 0 -T 0.3
# python main.py -datasets 'coco' -is_ft 1 -is_af 0 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0.2 -st 80 -wr 0 -T 0.3
python main.py -datasets 'coco' -is_ft 0 -is_af 1 -loss 'Focal' -is_gcn 0 -gcn_lr 1e-4 -rp 0.3 -st 80 -wr 0 -T 0.3
# python main.py -datasets 'coco' -is_ft 1 -is_af 0 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0.4 -st 80 -wr 0 -T 0.3
# python main.py -datasets 'coco' -is_ft 1 -is_af 0 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0.5 -st 80 -wr 0 -T 0.3
# python main.py -datasets 'coco' -is_ft 1 -is_af 0 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0.6 -st 80 -wr 0 -T 0.3
# python main.py -datasets 'coco' -is_ft 1 -is_af 0 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0.7 -st 80 -wr 0 -T 0.3
# python main.py -datasets 'coco' -is_ft 1 -is_af 0 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0.8 -st 80 -wr 0 -T 0.3
# python main.py -datasets 'coco' -is_ft 1 -is_af 0 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0.9 -st 80 -wr 0 -T 0.3
# python main.py -datasets 'coco' -is_ft 0 -is_af 1 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0.2 -st 80 -wr 0 -T 0.3

# python main.py -datasets 'coco' -is_ft 0 -is_vptd 1 -is_af 0 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0.2 -st 80 -wr 0 -T 0.3
# python main.py -datasets 'coco' -is_ft 0 -is_vptd 1 -is_af 1 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0.2 -st 80 -wr 0 -T 0.3
# python main.py -datasets 'coco' -is_ft 0 -is_vptsh 1 -is_af 0 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0.2 -st 80 -wr 0 -T 0.3
# python main.py -datasets 'coco' -is_ft 0 -is_vptsh 1 -is_af 1 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0.2 -st 80 -wr 0 -T 0.3
# python main.py -datasets 'coco' -is_ft 0 -is_vptd 1 -is_af 0 -is_prompt_tuning 0 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0.2 -st 80 -wr 0 -T 0.3
# python main.py -datasets 'coco' -is_ft 0 -is_af 0 -is_prompt_tuning 1 -loss 'DBLoss' -is_gcn 0
# python main.py -datasets 'coco' -is_ft 1 -is_af 0 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0.2 -st 80 -wr 1.0

# python main.py -datasets 'coco' -is_ft 1 -is_af 0 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0.2 -st 60 -T 0.1 # -is_gcn 0 -is_prompt 0 -is_ft 0 -is_af 1 -loss 'Focal' -lr 1e-4 
# python main.py -datasets 'coco' -is_ft 1 -is_af 0 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0.2 -st 60 -T 0.2
# python main.py -datasets 'coco' -is_ft 1 -is_af 0 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0.2 -st 60 -T 0.3
# python main.py -datasets 'coco' -is_ft 1 -is_af 0 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0.2 -st 60 -T 0.4
# python main.py -datasets 'coco' -is_ft 1 -is_af 0 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0.2 -st 60 -T 0.5
# python main.py -datasets 'coco' -is_ft 1 -is_af 0 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0.2 -st 60 -T 0.7
# python main.py -datasets 'coco' -is_ft 1 -is_af 0 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0.2 -st 60 -T 0.8
# python main.py -datasets 'coco' -is_ft 1 -is_af 0 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0 -st 60
# python main.py -datasets 'coco' -is_ft 1 -is_af 0 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0.2 -st 70
# python main.py -datasets 'coco' -is_ft 1 -is_af 0 -loss 'DBLoss' -is_gcn 1 -gcn_lr 1e-4 -rp 0 -st 70

# python main.py -datasets 'coco' -is_vptd 1 -rp 0.2 -st 60 -T 0.3 -is_gcn 1
# python main.py -datasets 'coco' -is_vptd 1 -is_af 0
# python main.py -datasets 'coco' -is_vptsh 1
# python main.py -datasets 'coco' -is_vptsh 1 -is_af 0
# python main.py -datasets 'voc' -rp 0.2 -st 20 -T 0.3 -is_gcn 0 -is_prompt 0 -is_ft 1 -is_af 0 -loss 'Focal'
# python main.py -datasets 'voc' -rp 0.2 -st 20 -T 0.3 -is_gcn 0 -is_prompt 0 -loss 'Focal'
# python main.py -datasets 'voc' -rp 0 -st 20 -T 0.3 -is_gcn 1 -is_prompt 1
# python main.py -datasets 'voc' -rp 0.2 -st 20 -T 0.3 -is_gcn 1 -is_prompt 1
# python main.py -datasets 'voc' -rp 0.2 -st 60 -T 0.3 -is_gcn 1 -is_prompt 1 -loss 'Focal'
# python main.py -datasets 'voc' -rp 0.2 -st 60 -T 0.3 -is_gcn 0 -is_prompt 0 -is_vptsh 1 -is_af 0 -loss 'Focal'
# python main.py -datasets 'voc' -rp 0.2 -st 60 -T 0.3 -is_gcn 0 -is_prompt 0 -is_vptd 1 -is_af 0 -loss 'Focal'

