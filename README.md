# Attention-free Language Models

This is the code accompanying my undergraduate thesis. We (1) compare and analyze attention-free language models' performance for sentiment analysis task; and (2) release UEH-ECOM, a 90000-sample dataset for e-commerce platform sentiment analysis. This work is published as a conference paper at ICICCT2024.

Supported models:
- RNN, LSTM, BiLSTM
- TextCNN
- gMLP
- Mamba

To run the code, clone this repository and run the following:

```
python main.py --train_file path/to/train.csv 
--test_file path/to/test.csv
--epochs 10
--batch_size 128
```

If you prefer to use Colab for reproducing the results, the link is available [here](https://colab.research.google.com/drive/11mA78WevQKLA7vt3OFcN0r6NnpryRM6X?usp=sharing).
