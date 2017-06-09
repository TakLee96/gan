
# Installation

### Experience with tensorflow on OSX

GPU support is extremely fragile on OSX, had multiple incompatibilities between CUDA / clang / tensorflow versions. After a full morning of tweaking still couldn't get it to work. I'm giving up with GPU support in the mac since the GPU in the macbook pro is anyway very limited in RAM and threads.

Now I plan to start using a linux machine (at INRIA) with multiple CPUs and GPUs. For this task, multiple CPUs is hopefully enough.

# Data preprocessing

Data preprocessing succeeded after some minor issues with python3. I solved these by switching temporarily to python 2 (fixing these is still useful and @ace, @jajiv seem to have been doing some work there).

# Inference

Command to train:

python run_summarization.py --mode=train --data_path="/sequoia/data3/fpedrego/dev/cnn-
dailymail/finished_files/chunked/train_*" --vocab_path=/sequoia/data3/fpedrego/dev/cnn-dailymail/finished_files/vocab --log_root=/sequoia/data3/fpedrego/data/pointer/ --exp_name=myexperiment


# Software versions
tensorflow==1.1.0
