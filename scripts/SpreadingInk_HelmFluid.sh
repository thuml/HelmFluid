python exp_real_video.py \
  --data-path /data/video/ \
  --ntrain 90 \
  --ntest 1 \
  --in_dim 30 \
  --out_dim 3 \
  --dataset_name real_video \
  --h 256 \
  --w 256 \
  --h-down 1 \
  --w-down 1 \
  --T-in 10 \
  --file_path real_1 \
  --T-out 10 \
  --groups 4 \
  --batch-size 2 \
  --learning-rate 0.0001 \
  --model HelmNet_2D_corr_potential_boundary \
  --d-model 64 \
  --patch-size 4,4 \
  --padding 0,0 \
  --model-save-path ./results/video \
  --model-save-name helmfluid_video_{}.pt