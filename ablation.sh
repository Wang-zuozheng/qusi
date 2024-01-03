export CUDA_VISIBLE_DEVICES='5'

python main.py -rp 0.2 -st 60 -T 0.3 -is_gcn 1 -is_prompt 1 -loss 'Focal'
python main.py -rp 0.2 -st 60 -T 0.3 -is_gcn 1 -is_prompt 1 -is_vptsh 1 -is_af 0
python main.py -rp 0.2 -st 60 -T 0.3 -is_gcn 1 -is_prompt 1 -is_vptd 1 -is_af 0
python main.py -rp 0.2 -st 60 -T 0.3 -is_gcn 1 -is_prompt 1 -is_ft 1 -is_af 0



