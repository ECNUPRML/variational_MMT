config = dict(    
    
    experiment = 'Multi30k', # Experiment
    lstm_hidden_units = 100, # Number of hidden units for the LSTM
    embedding_size = 300, # Word embedding dimension
    num_layers = 1, # Number of LSTM layers

    image_size = 48, # Shape of image

    encoder_vocab = 40000, # Vocabulary size on the encoder side # 30000 for dialogue
    decoder_vocab = 40000, # Vocabulary size on the decoder side # 30000 for dialogue
    encoder_num_tokens = 30, # Number of words/tokens in the input sequence # 20 for dialogue
    decoder_num_tokens = 20, # Number of words/tokens in the generated sequence

    dropout_keep_prob = 0.8, # Dropout keep probability
    initial_learning_rate = 0.005, # Initial learning rate
    learning_rate_decay = 0.75, # Learning rate decay
    min_learning_rate = 0.00001, # Minimum learning rate
    
    latent_dim = 100, # Dimension of z-latent space
    word_dropout_keep_probability = 0.75, # 1.0 - Word dropout rate for the decoder
    z_temp = 1.0, # Sampling temperature to be multiplied with the standard deviation

    batch_size = 100, # Batch size # 128 for dialogue
    n_epochs = 10, # Number of epochs

    logs_dir = 'summary_logs/var-seq2seq-det-attn', # Path to save summary information for Tensorboard
    model_checkpoint_dir = 'models/var-seq2seq-det-attn-', # Path to save model checkpoints
    bleu_path = 'bleu/var-seq2seq-det-attn', # Path to save model checkpoints
    w2v_dir = '../w2v_models/', # Word2Vec model directory
    data_dir = '../data/', # Directory to store data csv files

    load_checkpoint = 7, # Specify the trained model epoch/checkpoint number to be loaded for evaluation on test set, 0 means last saved checkpoint

)