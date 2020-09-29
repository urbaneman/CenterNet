cd src
# minicoco_dlaatt34_channel_conv1d_mul
# train
# python3.5 main.py ctdet --exp_id minicoco_dlaatt34_channel_conv1d_mul --arch dlaatt_34 --dataset minicoco --attention channel_conv1d_mul --batch_size 32 --master_batch 15 --lr 1.25e-4 --gpus 0,1

# test
python3.5 test.py ctdet --exp_id minicoco_dlaatt34_channel_conv1d_mul --arch dlaatt_34 --dataset minicoco --attention channel_conv1d_mul --keep_res --load_model ../exp/ctdet/minicoco_dlaatt34_channel_conv1d_mul/model_90.pth

# epoch 140

# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.277
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.439
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.293
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.117
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.304
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.397
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.258
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.416
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.437
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.246
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.470
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.616

# epoch 50
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.236
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.388
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.246
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.092
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.264
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.344
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.231
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.387
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.410
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.211
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.443
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.586

# epoch 90


# ctdet_minicoco_dlaatt_channel_conv1d_add
# train
# python3.5 main.py ctdet --exp_id minicoco_dlaatt34_channel_conv1d_add --arch dlaatt_34 --dataset minicoco --attention channel_conv1d_add --batch_size 32 --master_batch 15 --lr 1.25e-4 --gpus 0,1

# test
# python3.5 test.py ctdet --exp_id minicoco_dlaatt34_channel_conv1d_add --arch dlaatt_34 --attention channel_conv1d_add --keep_res --load_model ../exp/ctdet/minicoco_dlaatt34_channel_conv1d_add/model_last.pth

# epoch 140 5000|Tot: 0:02:22
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.283
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.447
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.298
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.118
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.314
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.411
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.264
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.422
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.443
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.247
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.475
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.612