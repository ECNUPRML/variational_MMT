
# Mutimodel Variational Sequence to Sequence Models


## Requirements
- tensorflow-gpu==1.3.0
- Keras==2.0. 8
- torchtext==0.6.0
- spacy==2.0.12
- numpy==1.12.1
- gensim==3.1.2
- nltk==3.4.5
- tqdm==4.19.1

## Instructions
1. Download spacy model
```
python -m spacy download en_core_web_md
python -m spacy download de_core_news_sm
``` 
2. Train the model, set configurations in the `model_config.py` file. For example,
```
cd ved_detAttn
vim model_config.py # Make necessary edits
python train.py
``` 
- The model checkpoints are stored in `models/` directory, the summaries for Tensorboard are stored in `summary_logs/` directory. As training progresses, the metrics on the validation set are dumped into`log.txt`  and `bleu/` directory.
3. Evaluate performance of the trained model. Refer to `predict.ipynb` to load desired checkpoint, calculate performance metrics (BLEU and diversity score) on the test set, and generate sample outputs.

