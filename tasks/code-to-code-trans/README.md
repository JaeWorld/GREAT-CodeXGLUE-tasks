### Fine-Tune
```
python run.py \
	--do_train \
	--do_eval \
	--model_type roberta \
	--model_name_or_path microsoft/codebert-base \
	--config_name roberta-base \
	--tokenizer_name roberta-base \
	--train_filename ../data/train.java-cs.txt.java,../data/train.java-cs.txt.cs \
	--dev_filename ../data/valid.java-cs.txt.java,../data/valid.java-cs.txt.cs \
	--output_dir ./saved_models \
	--max_source_length 512 \
	--max_target_length 512 \
	--beam_size 5 \
	--train_batch_size 16 \
	--eval_batch_size 16 \
	--learning_rate 5e-5 \
	--train_steps 30000 \
	--eval_steps 5000 \
            --load_model_path ../../../models/model_CD.bin \  # Modify this option to the name of the model you want to loa
            --output_model_name model_CD_CT.bin  > train_CD_CT.log 2>&1
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
	--train_filename ../data/train.java-cs.txt.java,../data/train.java-cs.txt.cs \
	--dev_filename ../data/valid.java-cs.txt.java,../data/valid.java-cs.txt.cs \
	--output_dir ./saved_models \
	--max_source_length 512 \
	--max_target_length 512 \
	--beam_size 5 \
	--train_batch_size 16 \
	--eval_batch_size 16 \
	--learning_rate 5e-5 \
	--train_steps 30000 \
	--eval_steps 5000 \
            --load_model_path ../../../models/model_DD_CS.bin \
	    --load_decoder_path ../../../models/model_CR.bin \   # use this option if you need to manually add decoder parameters 
            --output_model_name model_DD_CS_CT_kfold.bin  > train_DD_CS_CT_kfold.log 2>&1
```

### Run test loop (Test 10 models from k-fold cv)
```
python run_test_loop.py \
	--do_test \
	--model_type roberta \
	--model_name_or_path microsoft/codebert-base \
	--config_name roberta-base \
	--tokenizer_name roberta-base \
	--test_filename ../data/test.java-cs.txt.java,../data/test.java-cs.txt.cs \
	--output_dir ./saved_models \
	--max_source_length 512 \
	--max_target_length 512 \
	--beam_size 5 \
	--train_batch_size 16 \
	--eval_batch_size 16 \
	--learning_rate 5e-5 \
	--train_steps 30000 \
	--eval_steps 5000 \
            --model_dir ../../../models/model_DD_CS_CT_kfold  > test_DD_CS_CT_kfold.log 2>&1 
```
