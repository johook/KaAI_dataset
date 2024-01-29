CUDA_VISIBLE_DEVICES=0 python main_outside.py \
               --root_path ~/Driver-Intention-Prediction \
               --video_path_outside /home/jihwan/Driver-Intention-Prediction/datasets/annotation/road_camera \
               --annotation_path /home/jihwan/Driver-Intention-Prediction/datasets/annotation \
			   --result_path_outside outresults \
			   --pretrain_path_outside /home/jihwan/Driver-Intention-Prediction/outpt/convlstm.pth \
			   --dataset_outside Brain4cars_Outside \
			   --batch_size 8 \
			   --n_threads 4 \
			   --checkpoint 5  \
			   --n_epochs 1 \
			   --begin_epoch 1 \
			   --sample_duration 5 \
			   --end_second 5 \
			   --interval 5 \
			   --n_scales_outside 1 \
			   --learning_rate 0.1 \
			   --norm_value 255 \
			   --n_fold 0 \