### K-fold cross validation (k = 10)
```
python run_kfold.py \
	--do_train \
	--do_eval \
	--model_type roberta \
	--model_name_or_path microsoft/codebert-base \
	--config_name roberta-base \
	--tokenizer_name roberta-base \
	--train_filename ../data/small/train.buggy-fixed.buggy,../data/small/train.buggy-fixed.fixed \
	--dev_filename ../data/small/valid.buggy-fixed.buggy,../data/small/valid.buggy-fixed.fixed \
	--output_dir ./saved_models \
	--max_source_length 512 \
	--max_target_length 512 \
	--beam_size 5 \
	--train_batch_size 8 \
	--eval_batch_size 8 \
	--learning_rate 5e-5 \
	--train_steps 100000 \
	--eval_steps 5000
            --load_model_path ../../../models/model_DD_CS.bin \
            --output_model_name model_DD_CS_CT_kfold.bin  > train_DD_CS_CT_kfold.log 2>&1
```
