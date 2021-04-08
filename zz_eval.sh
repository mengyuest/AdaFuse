# eval AdaTSN50 model on SthV1
python -u main_gate.py something RGB --arch batenet50 --num_segments 8 --lr 0.01 --lr_steps 20 40 --epochs 50 --wd 5e-4 --npb --ada_reso_skip --init_tau 0.67 --gsmx --gate_history --hidden_quota 196608 --shared_policy_net --gbn --grelu --gate_gflops_loss_weight 0.125 --gflops_loss_type upb --batch-size 64 -j 72 --gpus 0 1 2 3 --exp_header X --test_from ada_fuse_pretrained/sthv1_adatsn50.pth.tar --skip_log | tee tmp_log.txt
OUTPUT0=`cat tmp_log.txt | tail -n 3`

# eval AdaTSM50 model on SthV1
python -u main_gate.py something RGB --arch batenet50 --num_segments 8 --lr 0.02 --lr_steps 20 40 --epochs 50 --wd 5e-4 --npb --ada_reso_skip --init_tau 0.67 --gsmx --gate_history --hidden_quota 196608 --shared_policy_net --gbn --grelu --gate_gflops_loss_weight 0.125 --gflops_loss_type upb --shift --batch-size 64 -j 72 --gpus 0 1 2 3 --exp_header X --test_from ada_fuse_pretrained/sthv1_adatsm50.pth.tar --skip_log | tee tmp_log.txt
OUTPUT1=`cat tmp_log.txt | tail -n 3`

# eval AdaTSM50Last model on SthV1
python -u main_gate.py something RGB --arch batenet50 --num_segments 8 --lr 0.02 --lr_steps 20 40 --epochs 50 --wd 5e-4 --npb --ada_reso_skip --init_tau 0.67 --gsmx --gate_history --gate_hidden_dim 1024 --gbn --grelu --gate_gflops_loss_weight 0.125 --gflops_loss_type upb --shift --batch-size 64 -j 72 --gpus 0 1 2 3 --exp_header X --enabled_stages 3 --test_from ada_fuse_pretrained/sthv1_adatsm50last.pth.tar --skip_log | tee tmp_log.txt
OUTPUT2=`cat tmp_log.txt | tail -n 3`

echo -e "\n\033[1;36mEXPECT Prec: 41.885 (sthv1_adatsn50)\033[0m"
echo $OUTPUT0
echo -e "\n\033[1;36mEXPECT Prec: 44.897 (sthv1_adatsm50)\033[0m"
echo $OUTPUT1
echo -e "\n\033[1;36mEXPECT Prec: 46.754 (sthv1_adatsm50last)\033[0m"
echo $OUTPUT2