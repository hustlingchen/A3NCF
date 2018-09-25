'''
Edit by Dr. Zhiyong Cheng based on Dr. Xiangnan He's NCF codes.
Added RMSE and MAE evaluation metrics for rating prediction.
The original code is of evaluating the performance of Top-K recommendation (Hit Ratio and NDCG for leave-1-out evaluation):
'''
import math
import keras
import heapq # for retrieval topK
import multiprocessing
import numpy as np
from time import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
#from numba import jit, autojit

# Global variables that are shared across processes
_model = None
_train = None
_testRatings = None
_testNegatives = None
_K = None


def eval_mae_rmse(model, testRating, user_review_fea, item_review_fea):
    _mae, _mse = [],[]
    #item_train = train.transpose()
    testU, user_fea, testI, item_fea, testL = [],[],[],[],[]
    
    for (u, i) in testRating.keys():
        # positive instance
        testU.append(u)
        user_fea.append(user_review_fea[u])
        testI.append(i)
        item_fea.append(item_review_fea[i])
        label = testRating[u,i]
        testL.append(label)
      
    #print testL[0:10];
    #one_hot_labels = keras.utils.to_categorical(testL, num_classes=5)  
    #print one_hot_labels[0:10,:]
    
    #print 'xxxxxxxxxxxxxxxxxxxxxxxxxxxx'
    score = model.predict([np.array(testU),  np.array(user_fea, dtype='float32'),np.array(testI), np.array(item_fea, dtype='float32')])
    #pred = np.argmax(score, axis=1)
    #pred = np.array(pred, dtype='float32')
    #print score[0:10]
    mae = mean_absolute_error(testL, score)
    mse = mean_squared_error(testL, score)
    rmse = np.sqrt(mse)
    return mae, rmse

def evaluate_model(model, train, testRatings, testNegatives, K, num_thread):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K
    global _train
    _model = model
    _train = train
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K
    
    hits, ndcgs = [],[]
    if(num_thread > 1): # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return (hits, ndcgs)
    # Single thread
    for idx in xrange(len(_testRatings)):
        (hr,ndcg) = eval_one_rating(idx)
        hits.append(hr)
        ndcgs.append(ndcg)      
    return (hits, ndcgs)

def test_generator(user, items):
    #t1 = time()
    users = np.full(len(items), user, dtype = 'int32')
    #users = np.array(users, dtype='int32')
    #items = np.array(items, dtype='int32')
    Xuser = _train[users, :].todense()
    #print user
    #print Xuser[0, :]
    Xitem = _train[:, items].todense().transpose()
    #print Xitem[0, :]
    #print Xuser.shape, Xitem.shape
    Xuser = np.array(Xuser, dtype='int32')
    Xitem = np.array(Xitem, dtype='int32')
    #t2 = time()
    #print('time %.1f s'%(t2-t1))
    return [Xuser, Xitem]
	
def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)
    # Get prediction scores
    map_item_score = {}
    #users = np.full(len(items), user_rep, dtype = 'int32')
    t1 = time()
    #users = np.full(len(items), u, dtype = 'int32')
    #predictions = _model.predict([_train[users,:].toarray(), _train.transpose()[items, :].toarray()],batch_size=100, verbose=0)
    predictions = _model.predict_on_batch(test_generator(u,items))
    #print('prediction time %.1f s' %(time()-t1))
    for i in xrange(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    
    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in xrange(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0
