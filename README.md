# GREAT-CodeXGLUE-tasks

These are the codes used for fine-tuning, kfold cv.
For each tasks, there are three files; run.py, run_kfold.py, run_test_loop.py

### Preparation
1. Clone the repository.
2. Follow the instructions on CodeXGLUE repository(https://github.com/microsoft/CodeXGLUE) to generate datasets. 
2. Place models in the `models` folder. All models are supposed to be in this folder.
3. Fine-tune models by running the script presented on each task.


### run.py
#### How to use
Let's suppose you want to fine-tune the model in the sequence of DD -> CD, and you already have model_DD.bin.
1. Place model_DD.bin in the `models` directory.
2. Modify the following options in the script as follows
    ```
    --model_name model_DD.bin 
    --output_model_name model_DD_CD.bin
    ```
3. Execute `run.py` file for the Clone detection task with the script.
