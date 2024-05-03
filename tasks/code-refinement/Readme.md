
### Fine-tune
```
python run.py \
	--do_train \
	--do_eval \
	--model_type roberta \
	--model_name_or_path microsoft/codebert-base \
	--config_name roberta-base \
	--tokenizer_name roberta-base \
	--train_filename ../data/train.buggy-fixed.buggy,../data/train.buggy-fixed.fixed \
	--dev_filename ../data/valid.buggy-fixed.buggy,../data/valid.buggy-fixed.fixed \
	--output_dir ./saved_models \
	--max_source_length 256 \
	--max_target_length 256 \
	--beam_size 5 \
	--train_batch_size 16 \
	--eval_batch_size 16 \
	--learning_rate lr=5e-5 \
	--train_steps 30000 \
	--eval_steps 5000
	--load_model_path ../../../models/model_CT.bin \
        --output_model_name model_CT_CR.bin  > train_CT_CR.log 2>&1
```

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
	--max_source_length 256 \
	--max_target_length 256 \
	--beam_size 5 \
	--train_batch_size 8 \
	--eval_batch_size 8 \
	--learning_rate 5e-5 \
	--train_steps 30000 \
	--eval_steps 5000 \
        --load_model_path ../../../models/model_CT_CD.bin \
        --output_model_name model_CT_CD_CR_kfold.bin  > train_CT_CD_CR_kfold.log 2>&1
```

### Run test loop (Test 10 models from k-fold cv)
```
python run_test_loop.py \
	--do_test \
	--model_type roberta \
	--model_name_or_path microsoft/codebert-base \
	--config_name roberta-base \
	--tokenizer_name roberta-base \
	--test_filename ../data/small/test.buggy-fixed.buggy,../data/small/test.buggy-fixed.fixed \
	--output_dir ./saved_models \
	--max_source_length 512 \
	--max_target_length 512 \
	--beam_size 5 \
	--train_batch_size 16 \
	--eval_batch_size 16 \
	--learning_rate 5e-5 \
	--train_steps 30000 \
	--eval_steps 5000 \
            --model_dir ../../../models/model_CD_DD_CT_CR > test_CD_DD_CT_CR_kfold.log 2>&1 &
 
```
