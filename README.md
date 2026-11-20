# transformer
transformer from scratch
Basically this is based on the ViT code, and I did some changes to make it a transformer."train" code is the one I used to train and "train+" is an improved version. As I have limited training resources, I only trained for 4000 steps and the result wasn't that perfect(it only learned the rough pattern of English and some words like "the") but I believe through training more steps, its performance can be improved.


Also, I implemented a BPE tokenizer while working on CS336 Assignment 1.

The implementation consists of three files:
* train_bpe_passed.py: Implements BPE training and generates the vocabulary and merge rules.
* implement_tokenizer.py: Implements the tokenizer, including encoding, decoding, and support for special tokens.
* timecounting.py:Profiles the BPE training process to measure the time spent on different operations and identify performance bottlenecks.
  
Results:
* Passed 23 tests, with 2 skipped due to macOS memory-testing limitations.
* Final encoding time: approximately 0.7 seconds for 30,000 characters
