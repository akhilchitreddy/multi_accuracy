#Code

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score, precision_score, accuracy_score
import json
import tensorflow as tf
from architecture import ANN
from sklearn.linear_model import Ridge
seed=128


class MultiModelBuilding:

    def __init__(self,path):
        self.path=path

    def res(p, y):
        return y * ((p>=0.1)/(p + 1e-20) + (p<0.1) * (20 - 100  * p)) +(1-y) * ((p < 0.9)/(1 - p + 1e-20) + (p>=0.9) * (100 * p - 80))

    def predict(self):
        learning_rate= 0.01
        training_epochs=4000
        def sess_run(result, x, sess,net):
            num = x.shape[0]

            num_batch = np.ceil(num/200).astype(int)

            output = np.zeros(num)
            for batch in range(num_batch):

                output[batch*200:(batch+1)*200] = sess.run(result, feed_dict={net.x:inputX_test[batch*200:(batch+1)*200],latent_ph:inputX_test[batch*200:(batch+1)*200]})
            return output

        data=pd.read_csv(self.path)
        data=data.rename(columns={'target':'target1'})
        data['target2']=1-data['target1']
        X_train,X_test,y_train,y_test=train_test_split(data.drop(['target1','target2'],axis=1),data[['target1','target2']],test_size=0.2,random_state=5)
        inputX=X_train.values
        inputY=y_train.values
        inputX_test=X_test.values
        inputY_test=y_test.values
        display_step=50
        n_samples=inputY.shape[0]
        net=ANN()
        init=tf.initialize_all_variables()
        sess=tf.Session()
        sess.run(init)
        for i in range(training_epochs):
            sess.run(net.optimizer,feed_dict={net.x:inputX,net.y_:inputY})
        pred=sess.run(net.y,feed_dict={net.x:inputX_test})
        control = tf.cast(tf.greater(net.y[:,1],net.y[:,0]), tf.float32)
        noharm = [control, 1 - control, control + 1 - control]
        logits = net.y[:,1] - net.y[:,0]
        max_T = 100
        thresh = 1e-4
        latent_ph = tf.placeholder(tf.float32, shape=(None, 124), name="latent_var")
        best_epoch, best_acc = -1,0
        #(idxs1, idxs2, _), _ = split_data(np.arange(len(idxs_val)), ratio=[0.7,0.3,0.])
        coeffs = []
        for t in range(max_T):
            control = tf.cast(tf.greater(net.y[:,1], net.y[:,0]), tf.float32)
            noharm = [control, 1 - control, control + 1 - control]
            probs_heldout = sess_run(tf.nn.sigmoid(logits), inputX_test[800:1600], sess=sess,net=net)
            heldout_loss = np.mean(-inputY_test[:,0][800:1600] * np.log(probs_heldout + 1e-20) - (1-inputY_test[:,0][800:1600]) * np.log(1-probs_heldout + 1e-20))
            heldout_acc =  np.mean((probs_heldout>0.5)==inputY_test[:,0][800:1600])
            probs = sess_run(tf.nn.sigmoid(logits), inputX_test,sess,net=net)
            val_loss = np.mean(-inputY_test[:,0] * np.log(probs + 1e-20) - (1 - inputY_test[:,0]) * np.log(1 - probs + 1e-20))
            val_acc = np.mean((probs > 0.5) == inputY_test[:,0])
            if heldout_acc > best_acc:
                best_epoch = t
                best_acc = heldout_acc
                best_logits = logits
            delta = MultiModelBuilding.res(probs,inputY_test[:,0])
            residual = probs - inputY_test[:,0]
            for i, s in enumerate(noharm):
                temp_s = sess_run(noharm[i], inputX_test[:800], sess,net=net)
                temp_s_heldout = sess_run(noharm[i], inputX_test[800:1600], sess,net=net)
                samples1 = np.where(temp_s == 1)[0]
                samples2 = np.where(temp_s_heldout == 1)[0]
                clf = Ridge(alpha=1)
                #clf.fit(inputX_test[:800],inputY_test[:800])
                clf.fit(inputX_test[:800],delta[:800])
                clf_prediction = clf.predict(inputX_test[800:1600])
                #corr = np.mean(clf_prediction[:,0] * residual[800:1600])
                corr = np.mean(clf_prediction * residual[800:1600])
                if corr > 1e-4:
                    coeffs.append(clf.coef_)
                    #h = (tf.matmul(tf.cast(inputX_test,tf.float32), tf.constant(np.expand_dims(clf.coef_,-1),
                     #                                     dtype=tf.float32)) + clf.intercept_)


                    h = (tf.matmul(latent_ph, tf.constant(np.expand_dims(clf.coef_,-1),
                                                          dtype=tf.float32)) + clf.intercept_)
                    h=tf.reshape(h,[-1])
                    logits -= .1 * h * s
                    #logits=tf.reshape(logits,[-1])
                    break
            if i==2:
                break
        probs = sess_run(net.y[:,1] - net.y[:,0], X_test, sess,net=net)
        groups=['all','race_asian','race_black','race_hispanic','race_native','race_other','race_white']
        #groups = ['all', 'F', 'M', 'B', 'N', 'BF', 'BM', 'NF', 'NM']
        errs = []
        idxs = list(range(0,1600))
        errs.append(100 * np.mean((probs[idxs]>0.5)!=y_test.iloc[idxs,0]))
        idxs = np.where((X_test['race_asian']==1))[0]
        #print(idxs)
        errs.append(100 * np.mean((probs[idxs]>0.5)!=y_test.iloc[idxs,0]))
        idxs = np.where((X_test['race_black']==1) )[0]
        errs.append(100 * np.mean((probs[idxs]>0.5)!=y_test.iloc[idxs,0]))
        idxs = np.where((X_test['race_hispanic']==1))[0]
        errs.append(100 * np.mean((probs[idxs]>0.5)!=y_test.iloc[idxs,0]))
        idxs = np.where((X_test['race_native']==1) )[0]
        errs.append(100 * np.mean((probs[idxs]>0.5)!=y_test.iloc[idxs,0]))
        idxs = np.where((X_test['race_other']==1) )[0]
        errs.append(100 * np.mean((probs[idxs]>0.5)!=y_test.iloc[idxs,0]))
        idxs = np.where((X_test['race_white']==1))[0]
        errs.append(100 * np.mean((probs[idxs]>0.5)!=y_test.iloc[idxs,0]))
      
        dict1={}
        metrics1={}
        for group, err in zip(groups, errs):
            dict1[group]=str(round(err, 1))
        ma_pred1=(probs<0.5).astype(int)
        actual=inputY_test[:,0]
        metrics1['accuracy'] = round(accuracy_score(actual, ma_pred1), 2)
        metrics1['model_recall'] = round(recall_score(actual, ma_pred1), 2)
        metrics1['model_precision'] = round(precision_score(actual, ma_pred1), 2)
        metrics1['roc_auc_score_model'] = round(roc_auc_score(actual,ma_pred1), 2)
        #print('Original: ', output)

        probs = sess_run(tf.nn.sigmoid(best_logits), X_test, sess,net=net)
        groups=['all','race_asian','race_black','race_hispanic','race_native','race_other','race_white']
        #groups = ['all', 'F', 'M', 'B', 'N', 'BF', 'BM', 'NF', 'NM']
        errs = []
        idxs = list(range(0,1600))
        errs.append(100 * np.mean((probs[idxs]>0.5)!=y_test.iloc[idxs,0]))
        idxs =np.where((X_test['race_asian']==1))[0]
        errs.append(100 * np.mean((probs[idxs]>0.5)!=y_test.iloc[idxs,0]))
        idxs = np.where((X_test['race_black']==1) )[0]
        errs.append(100 * np.mean((probs[idxs]>0.5)!=y_test.iloc[idxs,0]))
        idxs = np.where((X_test['race_hispanic']==1))[0]
        errs.append(100 * np.mean((probs[idxs]>0.5)!=y_test.iloc[idxs,0]))
        idxs = np.where((X_test['race_native']==1) )[0]
        errs.append(100 * np.mean((probs[idxs]>0.5)!=y_test.iloc[idxs,0]))
        idxs = np.where((X_test['race_other']==1) )[0]
        errs.append(100 * np.mean((probs[idxs]>0.5)!=y_test.iloc[idxs,0]))
        idxs = np.where((X_test['race_white']==1))[0]
        errs.append(100 * np.mean((probs[idxs]>0.5)!=y_test.iloc[idxs,0]))
        
        dict2={}
        metrics2={}
        for group, err in zip(groups, errs):
            dict2[group]=str(round(err, 1))
        ma_pred2=(probs<0.5).astype(int)
        metrics2['accuracy'] = round(accuracy_score(actual, ma_pred2), 2)
        metrics2['model_recall'] = round(recall_score(actual, ma_pred2), 2)
        metrics2['model_precision'] = round(precision_score(actual, ma_pred2), 2)
        metrics2['roc_auc_score_model'] = round(roc_auc_score(actual,ma_pred2), 2)


        before_ma = list(dict1.values())
        after_ma = list(dict2.values())
        metrics_before = list(metrics1.values())
        metrics_after = list(metrics2.values())
        for i in range(len(before_ma)):
            before_ma[i] = float(before_ma[i])
        for i in range(len(after_ma)):
            after_ma[i] = float(after_ma[i])
        for i in range(len(metrics_before)):
            metrics_before[i] = float(metrics_before[i]*100)
        for i in range(len(metrics_after)):
            metrics_after[i] = float(metrics_after[i]* 100)


        list_data=[
            {
                'name':'Before MultiAccuracy ',
                'data':dict1,
                'metrics':metrics1
                },
            {
                'name':'After MultiAccuracy',
                'data':dict2,
                'metrics':metrics2
                }
            ]
 
        return list_data

