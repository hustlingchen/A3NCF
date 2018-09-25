
# A^3NCF:An Adaptive Aspect Attention Model for Rating Prediction

This is our implementation of the paper:

Zhiyong Cheng, Ying Ding, Xiangnan He, Lei Zhu, Xuemeng Song, Mohan Kankanhalli: [A^3NCF: An Adaptive Aspect Attention Model for Rating  Prediction.](https://www.ijcai.org/proceedings/2018/0521.pdf). In IJCAI'18, Stockholm, Sweden, 2018.

<b>Please cite our IJCAI'18 paper if you use our codes. Thanks!</b>

Author: Dr. Zhiyong Cheng (https://sites.google.com/view/zycheng)

Codes and Enviorments
===
Our model is a two-step model: in the first step, a topic model is used to extract users and items' features from reviews; in the second step, the extracted features are integrated into an attentional neural network for rating prediciton

The implementation of topic model is in java, see "topic model" fold
The implementation of the attentional neural network is in Python with Keras (Version 2.1.2) using Theano (Version 0.9.0) as backend. The python code is implmented based on Dr. Xiangnan He's NCF code (https://github.com/hexiangnan/neural_collaborative_filtering)

### Topic Model
The implementation for our topic model is "userItemTopicModel.java"

To run the topic model, please check "tuningAspectNumberAndTopicNumber.java". Simply set the correct path of documents and pre-defined topic numbers (aspect number can also be set) and run. 

For the topic model training, we give an example dataset "Patio_Lawn_and_Garden" in the "data" fold: "index_Patio_Lawn_and_Garden.train.dat", "index_Patio_Lawn_and_Garden.val.dat", "index_Patio_Lawn_and_Garden.test.dat"

##### Data format: “userIndex \t\t itemIndex \t\t rating \t\t reviews”
 
Noted that (1) only the training data ".train.dat" is used in topic model for featrue extraction; (2) we proprecessed the reviews by removing stopwords and infrequent words.

The results of topic model are saved into "topicmodelresults": the file end with ".5.user.theta" is the extracted feature for users, where the "5" indicates the number of topics used is 5; similarly ".5.item.theta" is the extracted features for items. Besides, ".twords" are the top words for each topics. 

### python code
Examples to run the code can be found in the "run.py"
Run ancf.py:
```
 python acnf.py --dataset Patio_Lawn_and_Garden --k 5 --activation_function relu --epochs 300 --batch_size 256 --num_factors 5 --regs [0,0] --lr 0.0005 --learner adam --verbose 1 --out 0
```

Noted that "--k" indicates the number of topics in topic model, so make sure you have prepared ".k.user.theta" and ".k.item.theta" by using the topic model. "--num_factors" is the number of factors in the latent factor model. In our experiments, we simply set "k=num_factor". 

We public the splits and the extracted topic model  features (when k =5) of all the five datasets of Amazone used in our experiments. Please feel free to test. The splits and extracted features are in "python/data". Noted that the splits of users and items in the test and training are exactly the same as the ones in the training of "topic model". Just we did not include the "validation" dataset (i.e., ".val.dat") here. 

 
