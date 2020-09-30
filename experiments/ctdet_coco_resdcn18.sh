#cd src
# train
#python main.py ctdet --exp_id coco_resdcn18 --arch resdcn_18 --batch_size 114 --master_batch 18 --lr 5e-4 --gpus 0,1,2,3 --num_workers 16
# test
#python test.py ctdet --exp_id coco_resdcn18 --arch resdcn_18 --keep_res --resume
# flip test
#python test.py ctdet --exp_id coco_resdcn18 --arch resdcn_18 --keep_res --resume --flip_test
# multi scale test
#python test.py ctdet --exp_id coco_resdcn18 --arch resdcn_18 --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
#cd ..

cd src
# train
python3.5 main.py ctdet --exp_id minicoco_resdcn18 --arch resdcn_18 --dataset minicoco --batch_size 56 --master_batch 24 --lr 2.5e-4 --gpus 0,1 --num_workers 16
# test
python3.5 test.py ctdet --exp_id minicoco_resdcn18 --arch resdcn_18 --keep_res --resume
# flip test
python3.5 test.py ctdet --exp_id minicoco_resdcn18 --arch resdcn_18 --keep_res --resume --flip_test
# multi scale test
python3.5 test.py ctdet --exp_id minicoco_resdcn18 --arch resdcn_18 --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
cd ..

# test

# flip test
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.214
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.368
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.218
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.060
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.230
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.335
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.218
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.340
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.353
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.127
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.382
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.557

# multi scale test
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.250
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.416
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.259
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.087
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.259
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.378
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.237
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.383
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.404
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.182
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.436
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.602
