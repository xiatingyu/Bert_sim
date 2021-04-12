Bert_sim
====
This repository restore the sorce code and datasets for "Using Prior Knowledge to Guide BERT’s Attention in Semantic Textual Matching Tasks"

ESIM
----
1.First, you can use the `get_similarity.py` file in the `ESIM/scripts/preprocessing` folder to obtain the similarity matrix of the `msrp/sts/url dataset`. You can also use the matrix we have built, then this step can be skipped.<br>
2.Then run the `preprocess_msrp.py` file in the preprocessing folder to preprocess the data<br>
3.Run under the `ESIM/scripts/training` folder <br>
```python
main_msrp.py --proportion 0.1 --output 10
```
Proportion specifies the size of the data set, output determines the output model to the folder corresponding to different data sets, where 10 is 10% of the data<br>
You can run `python main_msrp.py` directly, and use the default 100% data at this time<br>

[Here](https://drive.google.com/file/d/1KshPlBu7StLaASJOBsXzp4HTTYzR75CS/view?usp=sharing) is the data we have processed, Please place it under the ESIM/data/dataset folder after downloading<br>

We use glove.840B.300d.txt as embedding, which can be downloaded [here](https://www.kaggle.com/takuok/glove840b300dtxt?select=glove.840B.300d.txt), and then please put it in the `ESIM\data\embeddings` folder<br>
