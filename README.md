Bert_sim
====
This repository restore the sorce code and datasets for "Using Prior Knowledge to Guide BERT’s Attention in Semantic Textual Matching Tasks"

ESIM
----
1.First, you can use the `get_similarity.py` file in the `ESIM/scripts/preprocessing` folder to obtain the similarity matrix of the **msrp/sts/url** dataset. For **QQP** dataset, you should use `get_similarity_quora.py`. You can also use the matrix we have built, then this step can be skipped.<br>

3.Then run the `preprocess_msrp.py` file in the preprocessing folder to preprocess the data.<br>

5.Run under the `ESIM/scripts/training` folder. <br><br>
```python
python main_msrp.py --proportion 0.1 --output 10
```
`--proportion` specifies the size of the dataset, `--output` determines the storage location of models trained on datasets of different sizes, where 10 is 10% of the data<br>
You can run `python main_msrp.py` directly, and use the default 100% data at this time.<br><br>

[Here](https://drive.google.com/file/d/1DZxRzZ6giKaZp6q5s44oogjrGLxCKu4N/view?usp=sharing) is the data we have processed, Please place it under the `ESIM/data/dataset` folder after downloading.<br>

We use **glove.840B.300d** as embedding, which can be downloaded [here](https://www.kaggle.com/takuok/glove840b300dtxt?select=glove.840B.300d.txt), and then please put it in the `ESIM\data\embeddings` folder.<br>


BERT
----
1.Download the data we have processed, and put it in the `UER/datasets` folder, Or you can use the `get_similarity.py` file in **ESIM** to preprocess the data.<br>

2.The data we provide cannot be directly used in the BERT model, so further preprocessing is required to adapt to the structure of the BERT model, use `get_similarity.py` in UER folder to further preprocess the data.<br>

3.Then use the following command to run the BERT model<br>
```python
python run.py --train_path datasets/msrp/train.tsv 
--dev_path datasets/msrp/dev.tsv 
--test_path datasets/msrp/test.tsv 
--output_model_path models/msrp_100.bin 
--proportion 1.0
```

