# -*- coding: utf-8 -*-
"""
Created on Fri May 11 18:31:41 2018

@author: infected
"""

import matplotlib.pyplot as plt
import progressbar 
import numpy as np
import tensorflow as tf
import gym
import time 

gym.logger.set_level(40)
env = gym.make('CartPole-v0')

#==============================================================================
# Fonctions 
#==============================================================================

def one_hot_encoding(scalar,dimension,min_val=0):    
    one_hot_vector = np.zeros([dimension,],dtype=int)
    one_hot_vector[scalar-min_val]=1
    return one_hot_vector
    
def one_hot_encoding_ndarray(scalar,dimension,min_val=0):
    num_scal = np.shape(scalar)[0]    
    one_hot_vector = np.zeros([num_scal,dimension],dtype=int)        
    for i in range(num_scal):
        one_hot_vector[i][int(scalar[i][0])-min_val]=1
    return one_hot_vector    

def one_hot_decoding(vector,dimension,min_val=0):    
    scalar = np.argmax(vector)+min_val
    return scalar

def create_dataset(simulation_datas):
    i = 0
#    features = np.zeros(np.shape(step))
#    labels = np.zeros(np.shape(step))
    print("processing dataset for training... could take a long time... please wait.")    
    
    for episode in simulation_datas:
        for step in episode[0]:    
            if i==0:
                data_formated = step
            else:    
                data_formated=np.vstack((data_formated,step))
            i = i+ 1
#                print(step)
            
#    features = data_formated[:,0:3]
#    labels = data_formated[:,4]
    return data_formated

def new_batch(features, labels, batch_size):
	i=0 
	temp_features = np.zeros([batch_size,features[0].shape[0]])
	temp_labels = np.zeros([batch_size,labels[0].shape[0]])
	
	num_datas = len(features)
	sorted_indexes = np.random.randint(0,num_datas-1,batch_size)
	
	for to_next_batch in sorted_indexes :
		temp_features[i] = features[to_next_batch]
		temp_labels[i] = labels[to_next_batch]
		i = i+1
		
	return temp_features,temp_labels    

#ici je normalise avec l'écart type, mais il est aussi possible
#de le faire avec l'écart min-max 
def normalize_datas(rawpoints, high=100.0, low=0.0):
    mean = np.sum(rawpoints,0)/np.shape(rawpoints)[1]
    std = np.std(rawpoints,0)
    
    return (rawpoints-mean)/std    
    
#==============================================================================
# Modèle 
#==============================================================================
####        A       ### vérifier que define fourni tout ce qu'il faut à train
##    FAIRE       ##### en return value 


def define_train_play_model(hidden_l_neurons, num_features, num_labels, alpha, 
                               x_train, y_train, number_of_training, batch_size,
                               x_test, y_test):
    
    dropout_rate = 0.9
    
    l1 = 16
    l2 = 32
    l3 = 64
    l4 = 32
    l5 = 16    
    
    number_of_param = (num_features+1)*l1+(l1+1)*l2*2+(l2+1)*l3*2+(l5+1)*num_labels    
    print("total number of parameters to train : ",number_of_param)    
    
    graph = tf.Graph()
    with graph.as_default():        
    
        with tf.name_scope('inputs'):
        	x = tf.placeholder(tf.float32, shape=[None,num_features],name='datas')
        	y_ = tf.placeholder(tf.float32, shape=	[None,num_labels],name='labels')
        
        with tf.name_scope('first_layer'):
            W1 = tf.Variable(tf.truncated_normal([num_features,l1], stddev=0.1),name='weights')
            b1 = tf.Variable(tf.truncated_normal([l1], stddev=0.1),name='biais')
            a1_ = tf.nn.relu(tf.matmul(x,W1)+b1)
            a1 = tf.nn.dropout(a1_, dropout_rate)
           
        with tf.name_scope('second_layer'):
            W2 = tf.Variable(tf.truncated_normal([l1,l2], stddev=0.1),name='weights')
            b2 = tf.Variable(tf.truncated_normal([l2], stddev=0.1),name='bias')
            a2_ = tf.nn.relu(tf.matmul(a1,W2)+b2)
            a2 = tf.nn.dropout(a2_, dropout_rate)
           
        with tf.name_scope('third_layer'):
            W3 = tf.Variable(tf.truncated_normal([l2,l3], stddev=0.1),name='weights')
            b3 = tf.Variable(tf.truncated_normal([l3], stddev=0.1),name='bias')
            a3_ = tf.nn.relu(tf.matmul(a2,W3)+b3)
            a3 = tf.nn.dropout(a3_, dropout_rate)
         
        with tf.name_scope('fourth_layer'):
            W4 = tf.Variable(tf.truncated_normal([l3,l4], stddev=0.1),name='weights')
            b4 = tf.Variable(tf.truncated_normal([l4], stddev=0.1),name='bias')
            a4_ = tf.nn.relu(tf.matmul(a3,W4)+b4)
            a4 = tf.nn.dropout(a4_, dropout_rate)
           
        with tf.name_scope('fifth_layer'):
            W5 = tf.Variable(tf.truncated_normal([l4,l5], stddev=0.1),name='weights')
            b5 = tf.Variable(tf.truncated_normal([l5], stddev=0.1),name='bias')
            a5_ = tf.nn.relu(tf.matmul(a4,W5)+b5)
            a5 = tf.nn.dropout(a5_, dropout_rate)
           
        with tf.name_scope('sixth_layer'):
            W6 = tf.Variable(tf.truncated_normal([l5,num_labels], stddev=0.1),name='weights')
            b6 = tf.Variable(tf.truncated_normal([num_labels], stddev=0.1),name='bias')
            y = tf.nn.softmax(tf.matmul(a5,W6)+b6)        
        

        with tf.name_scope('cross-entropy'):
        	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices = [1]))
        
        with tf.name_scope('training'):
#        	alpha = 0.03
            train_step = tf.train.GradientDescentOptimizer(alpha).minimize(cross_entropy)

        sess = tf.InteractiveSession()
        
        with tf.name_scope('parameters_initialization'):
        	tf.global_variables_initializer().run()
        #Chaque tour de boucle, on récupère aléatoirement 100 données d'entrainement
        #On fait le training avec une partie du training set : mini-batch gradient
        #Avec un seul à chaque fois : stochatstic gradient
        
        with progressbar.ProgressBar(max_value=number_of_training) as bar:
            print("\nstarting to train ::")
            
            refresh_loss_rate = int(number_of_training*0.1)
            loss_array = []
            for actual_iter in range(number_of_training):
                bar.update(actual_iter)

                x_batch, y_batch = new_batch(x_train, y_train, batch_size)
                _,loss = sess.run([train_step,cross_entropy], feed_dict={x: x_batch, y_:y_batch})
                loss_array.append(loss)
                plt.figure(1)
                if not actual_iter % refresh_loss_rate :
                    print("\nloss : {:.4f}\n ".format(loss))
                    plt.clf()
                    plt.plot(loss_array)
                    plt.title("loss")  
            
            with tf.name_scope('prediction'):
                correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
                     
                accuracy =  tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                
                print("\nmodel's accuracy (alpha=", alpha, ") :")
                print(sess.run(accuracy, feed_dict={x: x_test, y_: y_test}))
        
        env.reset()
        first_action = env.action_space.sample()
        observation, reward, done, info = env.step(first_action)
        input_step = observation.reshape(1,num_features)
        
        
        win=0
        total_reward_array = []
        for _ in range(100):
            env.reset()
            total_reward = 0            
            
            for _ in range(100):
    #            env.render()
    #            time.sleep(0.05) 
                action=one_hot_decoding(sess.run(y, feed_dict={x: input_step}),2)
                observation, reward, done, info = env.step(int(action.ravel()[0])) 
                total_reward += reward                
                input_step = observation.reshape(1,4)
                if done :
                    total_reward_array.append(total_reward)
#                    print(total_reward,"\n")
                    if (total_reward>50):
                        
                        win=win+1
                        
                    break
        
#        print(total_reward_array,"\n")
    print("total win game : ",win,"/n")    
    return total_reward_array

#
#def train_model(model, x_train, y_train, number_of_training, batch_size):
#    graph=model[0]
#    train_step = model[1]
#    
#    with tf.Session(graph=graph) as sess:    
#        with tf.name_scope('parameters_initialization'):
#            tf.global_variables_initializer().run()
#        
#        for actual_iter in range(number_of_training):
#            
#            x_batch, y_batch = new_batch(x_train, y_train, batch_size)
#            sess.run(train_step, feed_dict={x: x_batch, y_:y_batch})
    
         
#==============================================================================
# MAIN
#==============================================================================

if __name__ == '__main__':  #cette ligne permet, lors de l'import du fichier par un autre
                            #de ne pas exécuter le main


    #==============================================================================
    #Variables de simulation
#   ok_threshold = 50
    ok_episode = 0        
    reward_threshold = 50
    number_of_episode = 10000
    max_steps = 100
    #données d'entraiement
    
    input_datas =[]    
    #Format de la liste input_datas : 
    #[N=epsiodes] 
    #[0=episode_observations,1=total_reward]
    #[M=step]
    #[observation and previous_action]

    #==============================================================================
    
    t_initial = time.time()
    with progressbar.ProgressBar(max_value=number_of_episode) as bar:
        for i_episode in range(number_of_episode):
            bar.update(i_episode)
            env.reset()    
            #initialisation des variables d'episode
            t_episode = time.time() 
            total_reward = 0
            previous_action = 0
            previous_observation = [[0,0,0,0]]
            episode_observations = []
                
            for t in range(max_steps):        #nombre de steps de simulation
    #            env.render()
    #            time.sleep(0.01)
        
                action = env.action_space.sample() # take a random action
                observation, reward, done, info = env.step(action)        
        
                #dans episode observation, j'ai comme 
                                            #inputs : observations+previous action
                                            #output : next action
                episode_observations.append(np.append(previous_observation,[action,action]))        
                total_reward += reward
                #Si le jeu est perdu (i.e angle > 15°) alors done = True
                previous_action = action
                previous_observation = observation
                if done:
    #                print("Episode finished after {:d} timesteps".format(t+1))
                    if total_reward>reward_threshold:
                        ok_episode +=1
                        input_datas.append([episode_observations, total_reward])       
                        
                    break
    
#        print("episode simulation time : {:.2f} ms, dead at {} steps ".format((time.time()-t_episode)*1e3, t+1))
    print("Total simulation time : {:.3f} s".format(time.time()-t_initial,3))
    
    total =0    
    for step in input_datas:
        total = total + step[1]
    print("Average score : ", total/len(input_datas))
    
    formated_datas=create_dataset(input_datas)
    simulation_data_size=len(formated_datas)
    print("number of training datas:", simulation_data_size)
    #je récupère 80% des données de simulation pour entrainer mon réseau
    #les 20% restants sont pour la validation
    training_set = formated_datas[0:int(simulation_data_size*0.8),:]
    x_train=training_set[:,0:4]
    y_train=one_hot_encoding_ndarray(training_set[:,4:5],2)
    
    test_set = formated_datas[int(simulation_data_size*0.8)+1:,:]   
    x_test=test_set[:,0:4]
    y_test=one_hot_encoding_ndarray(test_set[:,4:5],2)
    
    total_reward_array=define_train_play_model(hidden_l_neurons=100, num_features=4, num_labels=2,
                                 x_train=x_train, y_train=y_train,
                                 alpha=0.001, number_of_training=10000, batch_size=100,
                                 x_test=x_test, y_test=y_test)
                                 
    print("Average score : " ,np.mean(total_reward_array))
    plt.figure(2)    
    plt.hist(total_reward_array, bins=np.arange(70))
    plt.show()
    
#    train_model(graph,training_set[:,0:4],training_set[:,5],1000,20)
    env.close()
