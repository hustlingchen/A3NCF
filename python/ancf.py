'''
Created on Sept 24, 2018

Keras Implementation of A^3NCF rating prediction model in:
CHEGN Zhiyong et al. A^3NCF: An Adaptive Aspect Attention Model for Rating Prediction, In IJCAI 2018. 
@author Zhiyong Cheng (jason.zy.cheng@gmail.com)

The code was developed based on Dr. Xiangnan He's NCF codes (https://github.com/hexiangnan/neural_collaborative_filtering).
@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import numpy as np
import theano.tensor as T
import keras
from keras import backend as K
from keras import initializers
from keras.models import Sequential, Model, load_model, save_model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.regularizers import l2
from Dataset import Dataset
from evaluate import eval_mae_rmse
from time import time
import multiprocessing as mp
import sys
import math
import argparse

#KERAS_BACKEND=tensorflow python -c "from keras import backend"
#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run GMF.")
    parser.add_argument('--path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--k', type=int, default=20,
                       help='Number of latent topics in represnetation')
    parser.add_argument('--activation_function', nargs='?', default='hard_sigmoid',
                       help='activation functions')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size.')
    parser.add_argument('--regs', nargs='?', default='[0,0]',
                        help="Regularization for user and item embeddings.")
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()

def init_normal(shape, name=None):
    return K.random_normal(shape, mean=0, stddev=0.01, seed = None)


from keras.utils.generic_utils import get_custom_objects

def clipped_relu(x):
    return K.relu(x, max_value=1)

def vallina_relu(x):   
    #return inputs * K.cast(K.greater(inputs, self.theta), K.floatx())
    #noise = K.random_normal((8,1), mean=0, stddev=0.00001)
    return K.cast(K.greater(x,0), K.floatx())+0.0001
   



def get_model(num_users, num_items, k, latent_dim, regs=[0,0], activation_function='hard_sigmoid'):
    #get_custom_objects().update({'vallina_relu': Activation(vallina_relu)})
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    user_fea = Input(shape=(k,), dtype='float32', name = 'user_fea')
    
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')
    item_fea = Input(shape=(k,), dtype='float32', name = 'item_fea')
    
    MF_Embedding_User = Embedding(embeddings_initializer=init_normal, name = 'user_embedding', output_dim = latent_dim, embeddings_regularizer = l2(regs[0]), input_dim = num_users,input_length=1)
    MF_Embedding_Item = Embedding(embeddings_initializer=init_normal, name = 'item_embedding', output_dim = latent_dim, embeddings_regularizer = l2(regs[0]),input_dim = num_items, input_length=1)   
    
    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MF_Embedding_User(user_input))
    item_latent = Flatten()(MF_Embedding_Item(item_input))
    user_latent = keras.layers.Add()([user_fea, user_latent])
    item_latent = keras.layers.Add()([item_fea, item_latent])
    user_latent = Dense(latent_dim, kernel_initializer='glorot_normal',  activation='relu')(user_latent)
    #user_latent = keras.layers.core.Dropout(0.2)(user_latent)
    item_latent = Dense(latent_dim, kernel_initializer='glorot_normal',  activation='relu')(item_latent)
    #item_latent = keras.layers.core.Dropout(0.2)(item_latent)
    # review-based attention calculation
    # user_side attention calculation
    #user_item_subtract = keras.layers.Concatenate()([user_fea, item_fea])
    #user_item_subtract.op.values = K.abs(user_item_subtract)
    user_item_concat = keras.layers.Concatenate()([user_fea, item_fea, user_latent, item_latent])
    att = Dense(latent_dim, kernel_initializer='random_uniform',  activation='softmax')(user_item_concat)
    #att_soft = Dense(latent_dim, kernel_initializer=init_normal, kernel_regularizer=keras.regularizers.l2(0.0), activation='softmax', use_bias=False)(att)
    #uatt.op.values = K.softmax(uatt)
    vec = keras.layers.Multiply()([user_latent, item_latent])
    
    

   
    # Element-wise product of user and item embeddings 
    predict_vec = keras.layers.Multiply()([vec, att])
   
     #Final prediction layer
    #prediction = Lambda(lambda x: K.sigmoid(K.sum(x)), output_shape=(1,))(predict_vector)
    prediction = Dense(latent_dim, kernel_initializer='glorot_normal',  activation='relu')(predict_vec)
    prediction = keras.layers.core.Dropout(0.5)(prediction)
    prediction = Dense(1, kernel_initializer='glorot_normal', name = 'prediction')(prediction)
    
    model = Model(inputs=[user_input, user_fea, item_input, item_fea], outputs=prediction)

    return model

def get_train_instances(train, user_review_fea, item_review_fea):
    user_input, user_fea, item_input, item_fea, labels = [],[],[],[],[]
    num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        user_fea.append(user_review_fea[u])
        item_input.append(i)
        item_fea.append(item_review_fea[i])
        label = train[u,i]
        labels.append(label)
    #one_hot_labels = keras.utils.to_categorical(labels, num_classes=5)  
     
        
    return user_input, np.array(user_fea, dtype='float32'), item_input, np.array(item_fea, dtype='float32'), labels

if __name__ == '__main__':
    args = parse_args()
    num_factors = args.num_factors
    k = args.k
    regs = eval(args.regs)
    learner = args.learner
    learning_rate = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    verbose = args.verbose
    activation_function = args.activation_function
    
    evaluation_threads = 1 #mp.cpu_count()
    print("A3NCF arguments: %s" %(args))
    #model_out_file = 'Pretrain/%sNumofTopic_%d_GMF_%d_%d.h5' %(args.dataset, k, num_factors, time())
    
    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset,k)
    train, user_review_fea, item_review_fea, testRatings = dataset.trainMatrix, dataset.user_review_fea, dataset.item_review_fea, dataset.testRatings
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
          %(time()-t1, num_users, num_items, train.nnz, len(testRatings)))
    
    # Build model
    model = get_model(num_users, num_items, k, num_factors, regs, activation_function)
    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='mean_squared_error')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='mean_squared_error')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='mean_squared_error')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='mean_squared_error')
    #print(model.summary())
    
    # Init performance
    t1 = time()
    (mae, rmse) = eval_mae_rmse(model, testRatings, user_review_fea, item_review_fea)
    print('Init: MAE = %.4f, RMSE = %.4f\t [%.1f s]' %(mae, rmse, time()-t1))

    
    # Train model
    best_mae, best_rmse, best_iter = mae, rmse, -1

    for epoch in xrange(epochs):
        t1 = time()
        # Generate training instances
        user_input, user_fea, item_input, item_fea, labels = get_train_instances(train, user_review_fea, item_review_fea)
        
        # Training
        hist = model.fit([np.array(user_input), user_fea, np.array(item_input), item_fea], #input
                         np.array(labels), # labels 
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        t2 = time()
        
        # Evaluation
        if epoch %verbose == 0:
            (mae, rmse) = eval_mae_rmse(model, testRatings, user_review_fea, item_review_fea)
            loss = hist.history['loss'][0]
            print('Iteration %d [%.1f s]: mae = %.4f, rmse = %.4f, loss = %.4f [%.1f s]' 
                  % (epoch,  t2-t1, mae, rmse, loss, time()-t2))
            if rmse < best_rmse:
                best_mae, best_rmse, best_iter = mae, rmse, epoch
                #if args.out > 0:
                #    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  mae = %.4f, rmse = %.4f. " %(best_iter, best_mae, best_rmse))
    #if args.out > 0:
    #    print("The best ancf model is saved to %s" %(model_out_file))
    
    outFile = 'results/ancf' +  '.result'
    f = open(outFile, 'a')
    f.write(args.dataset + '\t' + activation_function + "\t" + str(num_factors)+ '\t' + str(best_mae) + '\t' + str(best_rmse) + '\n');
    f.close();
    
   
