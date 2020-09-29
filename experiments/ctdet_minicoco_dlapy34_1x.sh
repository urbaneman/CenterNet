cd src
# minicoco_dlaatt34_channel_conv1d_mul
# train
# python3.5 main.py ctdet --exp_id minicoco_dlapy34 --arch dlapy_34 --dataset minicoco --pyconv True --batch_size 32 --master_batch 15 --lr 1.25e-4 --gpus 0,1

# test
# python3.5 test.py ctdet --exp_id minicoco_dlapy34 --arch dlapy_34 --dataset minicoco --pyconv True --keep_res --load_model ../exp/ctdet/minicoco_dlapy34/model_last.pth

# epoch 140
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.281
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.445
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.122
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.305
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.400
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.261
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.421
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.441
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.251
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.469
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.615
