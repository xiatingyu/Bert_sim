Bert_sim
====
This repository restore the sorce code and datasets for "Using Prior Knowledge to Guide BERT’s Attention in Semantic Textual Matching Tasks"

ESIM
----
1.First, you can use the get_similarity.py file in the scripts/preprocessing folder to obtain the similarity matrix of the msrp/sts/url dataset. You can also use the matrix we have built, then this step can be skipped.
2.Then run the preprocess_msrp.py file in the preprocessing folder to preprocess the data
3.Run under the scripts/training folder python main_msrp.py --proportion 0.1 --output 10
Proportion specifies the size of the data set, output determines the output model to the folder corresponding to different data sets, where 10 is 10% of the data
You can run python main_msrp.py directly, and use the default 100% data at this time

[Here](https://drive.google.com/file/d/1KshPlBu7StLaASJOBsXzp4HTTYzR75CS/view?usp=sharing) is the data we have processed, Please place it under the ESIM/data/dataset folder after downloading
