cd src
# train
python main.py ctdet --exp_id minicoco_resdcn18_gc --arch resdcnmff_18 --attention channel_add --dataset minicoco --batch_size 56 --master_batch 24 --lr 2.5e-4 --gpus 0,1 --num_workers 16
# test
python test.py ctdet --exp_id minicoco_resdcn18_gc --arch resdcnmff_18 --attention channel_add --dataset minicoco --keep_res --resume
# flip test
python test.py ctdet --exp_id minicoco_resdcn18_gc --arch resdcnmff_18 --attention channel_add --dataset minicoco --keep_res --resume --flip_test
# multi scale test
python test.py ctdet --exp_id minicoco_resdcn18_gc --arch resdcnmff_18 --attention channel_add --dataset minicoco --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
cd ..