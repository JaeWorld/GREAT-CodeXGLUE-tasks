### Fine-Tune
```
python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_train \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --load_custom_model \  # This option is necessary if you want to load pretrained model
    --model_name model_CD.bin \   # Modify this option to the name of the model you want to load
    --output_model_name model_CD_DD.bin \
    --seed 123456  2>&1 | tee train_CD_DD.log
```

### K-fold cross validation (k = 10)
```
python run_kfold.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_train \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --load_custom_model \
    --model_name model_CT_CS.bin \
    --output_model_name model_CT_CS_DD_kfold.bin \
    --seed 123456  2>&1 | tee train_CT_CS_DD_kfold.log
```

### Run test loop (Test 10 models from k-fold cv)
```
python run_test_loop.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_eval \
    --do_test \
    --train_data_file=../dataset/train.jsonl \
    --eval_data_file=../dataset/valid.jsonl \
    --test_data_file=../dataset/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --model_dir ../../../models/model_CS_CT_DD_kfold \
    --seed 123456 2>&1 | tee test_CS_CT_DD_kfold.log
```
