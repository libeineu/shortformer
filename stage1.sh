device=0,1,2,3,4,5,6,7
# device=0
gpu_num=8


data=wiki
tag=pre_cor_v5_shorformer_Layer16_stage1_v2

if [ $data == "wiki" ]; then
        arch=numerical_transformer_v5_lm_wiki103
        max_tokens=2560
        tokens_per_sample=128
        update_freq=1
        dropout=0.3
        weight_decay=0
        keep_last_epochs=200
        criterion=adaptive_loss
        max_epoch=
        max_update=140100
        data_dir=wikitext-103
        fp16=1
elif [ $data == "ptb" ]; then
        arch=transformer_lm_single
        lr=0.0007
        warmup=2000
        max_tokens=4096
        tokens_per_sample=${max_tokens}
        update_freq=1
        criterion=label_smoothed_cross_entropy
        dropout=0.1
        weight_decay=0.01
        keep_last_epochs=2
        max_epoch=20
        max_update=
        data_dir=penn
        adam_betas="'(0.9, 0.997)'"
        fp16=1
else
        echo "unknown task=$data"
        exit
fi
#tag=$arch"_"${max_tokens}"_"${dropout}"_"${lr}
save_dir="checkpoints/lm/${data}/$tag"

if [ ! -d $save_dir ]; then
        mkdir -p $save_dir
fi
cp ./stage1.sh $save_dir/train_lm.sh


if [ $data == "wiki" ]; then
    cmd="python -m torch.distributed.launch --nproc_per_node=8
        --nnodes=4 --node_rank=0 --master_addr="172.16.1.117"
	--master_port=20202
        train.py data-bin/wikitext-103
        --task language_modeling          
        --save-dir $save_dir     
        --arch $arch
        --decoder-layers 8
        --dec-calculate-num 2
        --dec-learnable-type ema     
        --max-update 140100 
        --max-lr 1.0 
        --t-mult 2 
        --lr-period-updates 270000 
        --lr-scheduler cosine 
        --lr-shrink 0.75     
        --warmup-updates 16000 
        --warmup-init-lr 1e-07 
        --min-lr 1e-09 
        --optimizer nag 
        --lr 0.0001 
        --clip-norm 0.1     
        --criterion adaptive_loss 
        --max-tokens $max_tokens 
        --update-freq $update_freq
        --seed 1 
        --fp16     
        --sample-break-mode none 
        --skip-invalid-size-inputs-valid-test 
        --ddp-backend=no_c10d 
        --tokens-per-sample 128 
        --max-tokens-valid 128 
        --tokens-from-prev 128 
        --curriculum 1000 
        --required-batch-size-multiple 1 
        --save-interval 5"
elif [ $data == "ptb" ]; then
     cmd="python3 -u train.py data-bin/$data_dir
        --optimizer adam
        --clip-norm 0.0
        --lr-scheduler inverse_sqrt
        --warmup-init-lr 1e-07
        --min-lr 1e-09
        --label-smoothing 0.1
        --no-progress-bar
        --log-interval 100
        --ddp-backend no_c10d
        --task language_modeling
        --distributed-world-size $gpu_num
        --criterion ${criterion}
        --arch $arch
        --weight-decay $weight_decay
        --warmup-updates $warmup
        --lr $lr
        --max-tokens $max_tokens
        --dropout $dropout
        --update-freq $update_freq
        --save-dir $save_dir
        --keep-last-epochs $keep_last_epochs
        --tensorboard-logdir $save_dir"
      cmd=${cmd}" --adam-betas "${adam_betas}
else
        echo "unknown task=$data"
        exit
fi

if [ $fp16 -eq 1 ]; then
cmd=${cmd}" --fp16 "
fi

export CUDA_VISIBLE_DEVICES=$device
cmd="nohup "${cmd}" > $save_dir/train.log 2>&1 &"
eval $cmd
tail -f $save_dir/train.log


