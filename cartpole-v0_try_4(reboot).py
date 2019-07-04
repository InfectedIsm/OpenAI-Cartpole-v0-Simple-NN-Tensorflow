# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 21:54:57 2018

@author: infected
"""
import sys
#needed to calculate time elapsed between simulations or training
#also useful to pause de gym environment in order to be able to
#look at the animation
import time 
##progress bar that indicate percentage of work done
import progressbar
#in order to trace different plots
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import gym

#this global variable activate the debugprint function
#but if the 'override' parameter is set to true, even if DEBUG=False
#the string is printed
DEBUG = False

#=============================================================================================================================================
#=================================Useful functions=====================================================================================
#=============================================================================================================================================


#if the 'override' parameter is set to true, even if DEBUG=False
#the string is printed
def debugprint(s, override=False):
    if DEBUG or override:
        print(s)

def one_hot_decoding(vector,dimension,min_val=0):    
    scalar = np.argmax(vector)+min_val
    return scalar
    
    
def new_batch(features, labels, batch_size):
    if type(features) == list and type(labels) == list:
        temp_features = np.zeros([batch_size,np.array(features[0]).shape[0]])
        temp_labels = np.zeros([batch_size,np.array(labels[0]).shape[0]])
        
        
    elif type(features) == np.ndarray and type(labels) == np.ndarray:
        temp_features = np.zeros([batch_size,features[0].shape[0]])
        temp_labels = np.zeros([batch_size,labels[0].shape[0]])
        
    else :
        raise AttributeError('Error : accepted types : \'list\' or \'np.ndarray\' features and labels types must be the same')
        
    num_datas = len(features)
    random_indexes = np.random.randint(0,num_datas-1,batch_size)
    
    for i,to_next_batch in enumerate(random_indexes) :
        temp_features[i] = features[to_next_batch]
        temp_labels[i] = labels[to_next_batch]

    return temp_features,temp_labels

#=============================================================================================================================================
#=================================Environment definition=====================================================================================
#=============================================================================================================================================

gym.logger.set_level(40)
env = gym.make('CartPole-v0')


#============================================== 
# Generate training datas by playing a lot of games in the environment
# Only the best games are kept as training datas
#============================================== 

def generate_training_datas(number_of_episodes, reward_threshold=50, max_step=100, last_steps_to_del=20, trace_plot=True) :

    try:    
        ok_episode = 0        
        t_initial = time.time()
        training_datas = []
        all_scores = []
        accepted_scores=[]
        
        time.sleep(0.1) #without this pause, there is a bug with the progressbar
        with progressbar.ProgressBar(max_value=number_of_episodes) as bar:    
            for i_episode in range(number_of_episodes):
                bar.update(i_episode)
                #reset l'environnement pour recommencer le jeu à 0
                env.reset()    
                #initialisation des variables d'episode
#                t_episode = time.time() 
                total_reward = 0            
                game_datas = []
                prev_observation = []
                
                for t in range(max_steps):           
#                    env.render()
#                    time.sleep(0.01)
                    action = env.action_space.sample() # take a random action in action space
                    observation, reward, done, info = env.step(action)
                    if len(prev_observation)>0:           
                        game_datas.append([prev_observation,action])
                    prev_observation = observation
                    total_reward += reward
                    
                    if done: break #Si le jeu est perdu (i.e angle > 15°) alors done = True
                
                all_scores.append(total_reward)
                    
                if total_reward >= reward_threshold:
                    ok_episode+=1
                    accepted_scores.append(total_reward)
                    #one_hot_encoding right below
                    for action in game_datas:
                        if action[1] == 0:
                            action[1] = [1,0]
                        else:
                            action[1] = [0,1]   
                    training_datas.append(game_datas[0:len(game_datas)-last_steps_to_del]) #remove the last obs and action because they make the game lose
        
        #        print("episode simulation time : {:.2f} ms, dead at {} steps ".format((time.time()-t_episode)*1e3, t+1))
        print(">> Total simulation time : {:.3f} s".format(time.time()-t_initial,3))
        print(">> Number of ok episodes : "+str(ok_episode))
        number_of_ok_steps = np.shape(training_datas)[0]*np.shape(training_datas)[1]
        print(">> Number of steps ok for training : "+str(number_of_ok_steps))
        
        if trace_plot:        
            plt.figure(1)
            plt.hist(all_scores, bins=np.arange(150), label="all_scores")
            plt.hist(accepted_scores, bins=np.arange(150), label="acepted_scores")
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)
            plt.show()
        
        env.close()
        
    except Exception as e:
        print('\nunindentified error happened (line {}) : \n{}'.format(sys.exc_info()[-1].tb_lineno,str(e)))
        env.close()
    
    return training_datas

#============================================== 
#  Use the generated neural network to play the game
#============================================== 

def play(number_of_games, max_steps=200, win_limit=100, render=False, debug = DEBUG):
    win=0        
    play_scores = []    
    env.reset()    
    
    with graph.as_default():
        y = tf.get_default_graph().get_tensor_by_name("layer-6/output:0")
        first_action = env.action_space.sample()    
        observation, reward, done, info = env.step(first_action)
        input_step = observation.reshape(1,4)
        
        time.sleep(0.1) #without this pause, there is a bug with the progressbar
        with progressbar.ProgressBar(max_value=number_of_games) as bar:
            for game in range(number_of_games):
                bar.update(game)
                total_reward = 0            
                env.reset()    
        
                for step in range(max_steps):
                    if render:
                        env.render()
                        time.sleep(0.01)
    
                    if (debug and game==0 and step ==0) :     
                        t = time.time()
                        
                    action=one_hot_decoding(sess.run(y, feed_dict={"inputs/datas:0": input_step}),2)
    
                    if (debug and game==0 and step ==0) :     
                        print("\n>> inference time : ",time.time()-t)
    
                    observation, reward, done, info = env.step(int(action.ravel()[0])) 
                    total_reward += reward                
                    input_step = observation.reshape(1,4) 
    
                    if (debug and game==0 and step ==0) : 
                        print(">> step duration : ",time.time()-t)
                        
                    if done : break
                        
                play_scores.append(total_reward)
                
                if (total_reward>win_limit):                
                    win=win+1                        
     
    env.close()           
    
#        print(play_scores,"\n")
    print(">> percentage of won games (>={:d} steps) : ".format(win_limit),win/number_of_games*100,"%")  
    
    plt.figure(3)
    limit_low = min(play_scores)-int(min(play_scores)*0.2)
    limit_high = max(play_scores)+int(max(play_scores)*0.2)
    plt.hist(play_scores, bins=np.arange(limit_low,limit_high), label="play_scores")
    plt.show()
    
    max_score = max(play_scores)
    min_score = min(play_scores)
    mean_score = np.mean(play_scores)
    
    print("\n>> max score :",max_score)
    print(">> min score :",min_score)
    print(">> mean score :",mean_score)
    
    return play_scores


#=============================================================================================================================================
#=================================Neural Network Definition===================================================================================
#=============================================================================================================================================

class Model :
    def __init__(self, learning_rate, dropout_rate, io_size, hidden_layer_sizes):
        self.learning_rate = learning_rate
        self.dropout_rate = 1- dropout_rate
        self.io_size = io_size
        self.hidden_layer_sizes = hidden_layer_sizes


#============================================== 
#  Generate a layer, used in define_model() right below 
#============================================== 

def generate_layer(dimensions, input_tensor, last=False, dropout_rate=0.85):
    W = tf.Variable(tf.truncated_normal([dimensions[0],dimensions[1]], stddev=0.1),name='weights')
    b = tf.Variable(tf.zeros([dimensions[1]]),name='biais')
    if not last:    
        a_ = tf.nn.relu(tf.matmul(input_tensor,W)+b, name='output')
        a = tf.nn.dropout(a_, dropout_rate)
    else:
        a = tf.nn.softmax(tf.matmul(input_tensor,W)+b, name='output')
    return a

#============================================== 
#  This part generate the graph
#  For now, it dont take any parameters, but I aim to give it
# parameters and generate a model from them
#============================================== 

def define_model(model, debug=False):
    tf.reset_default_graph() 
    graph = tf.Graph()
    
    alpha = model.learning_rate
    dropout_rate = model.dropout_rate
    num_inputs = model.io_size[0]
    num_labels = model.io_size[1]
    layers = model.hidden_layer_sizes
    
    #create a vector with elements [num_inputs, layers, num_labels]
    all_layers = np.insert( np.insert( np.array(layers),len(layers),num_labels), 0, num_inputs) 
    if debug:    
        print(all_layers)
    
    #calculate the total number of parameters in the network
    all_layers_1 = all_layers + 1
    all_layers = np.insert(all_layers,len(all_layers),0)
    all_layers_1 = np.insert(all_layers_1,0,0)
    parameters_to_train = all_layers * all_layers_1
    parameters_to_train = np.sum(parameters_to_train)
    
    
    print('\n>> parameters to train : {}'.format(parameters_to_train))
    print('please wait ~1mn if no progress bar appear')
    
    layers.append(layers[len(layers)-1]) #needed to generate the last layer
        
    
    prev_layer_size = 0
    
    with graph.as_default():
        
        with tf.name_scope('inputs'):
        	x = tf.placeholder(tf.float32, shape=[None,num_inputs],name='datas')
        	y_ = tf.placeholder(tf.float32, shape=	[None,num_labels],name='labels')
            
        for layer_index,layer_size in enumerate(layers):
            layer_name = "layer-"+str(layer_index+1)
            if debug:            
                print(layer_name, end='')
    
            with tf.name_scope(layer_name):
                if layer_index == 0:
                    debugprint(" in"+str([num_inputs,layer_size]), override=debug)
                    layer_output=generate_layer([num_inputs,layer_size], x,dropout_rate=dropout_rate)
                
                elif layer_index == len(layers)-1:
                    debugprint(" out"+str([layer_size,num_labels]), override=debug)
                    y=generate_layer([prev_layer_size,num_labels], layer_output, last=True)
                    
                else:           
                    debugprint(" central"+str([prev_layer_size,layer_size]), override=debug)                
                    layer_output=generate_layer([prev_layer_size,layer_size], layer_output, dropout_rate=dropout_rate)
                    
            prev_layer_size = layer_size
            
        
        with tf.name_scope('categoricalCrossEntropy'):
        	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), 
        						reduction_indices = [1]), name='crossEntropy')
                            
        with tf.name_scope('training'):
            alpha = 0.03
            train_step = tf.train.GradientDescentOptimizer(alpha).minimize(cross_entropy)

#        for op in tf.get_default_graph().get_operations():
#            print("["+str(op.name)+"]")      
    
        with tf.name_scope('prediction'):		
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            accuracy =  tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')             
    
    return graph

#==============================================  
##  Just open a new session and return it
#==============================================

def open_session(graph):    
    
    with graph.as_default():    
#        all_variables_list = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
#        init = tf.variables_initializer(var_list=all_variables_list)
        init=tf.global_variables_initializer()
        
    session = tf.Session(graph=graph)
    session.run(init)
    tf.summary.FileWriter("tensorboard_test",session.graph)
    return session


#==============================================  
##  Train the network with as input a graph, and datas/labels
#==============================================


def training_run(graph, sess, x_train, y_train, number_of_training=1000,batch_size=100):
    with graph.as_default():
        
        loss_array = []
        smoothed_loss_array = []        
#        refresh_loss_rate = number_of_training*0.1
        train_step=tf.get_default_graph().get_operation_by_name("training/GradientDescent")
        accuracy=tf.get_default_graph().get_tensor_by_name("prediction/accuracy:0")    
        
        time.sleep(0.2) #without this pause, there is a bug with the progressbar
        with progressbar.ProgressBar(max_value=number_of_training) as bar:
            for actual_iter in range(number_of_training):
                bar.update(actual_iter)
                #create a batch of datas from train_datas vector
                x_batch, y_batch = new_batch(x_train, y_train, batch_size)                
                #get cross_entropy tensor from graph in order to use it
                cross_entropy = tf.get_default_graph().get_tensor_by_name("categoricalCrossEntropy/crossEntropy:0")
                                                
                _,loss,training_accuracy = sess.run([train_step,cross_entropy,accuracy], feed_dict={"inputs/datas:0": x_batch, "inputs/labels:0":y_batch})
                loss_array.append(loss)
                if len(loss_array)>20:
                    smoothed_loss_array.append(np.sum(loss_array[len(loss_array)-20:len(loss_array)])/20.)
                    
#                if not actual_iter % refresh_loss_rate :
##                    print("\nloss : {:.4f}\n ".format(loss))  
#                    plt.plot(loss_array)
#                    plt.title("loss")
#                    time.sleep(0.01)
            plt.figure(2)
            plt.plot(loss_array)            
            plt.plot(smoothed_loss_array, 'c')            
            plt.title("loss")
            axes = plt.gca()
            limit_low = min(loss_array)-min(loss_array)*0.1
            limit_high = max(loss_array)+max(loss_array)*0.1
            axes.set_ylim([limit_low,limit_high])    
            plt.show()
            print("last loss : {:.4f}".format(loss))
            print("training accuracy : {:.4f}".format(training_accuracy))
                        
    return loss,training_accuracy


#==============================================  
##  Test the network accuracy
#==============================================

def validation_run(sess, x_test, y_test):        
    with graph.as_default():    
        accuracy=tf.get_default_graph().get_tensor_by_name("prediction/accuracy:0")    
    print("\n>> accuracy is being calculated... please wait few seconds")
    validation_accuracy = sess.run(accuracy, feed_dict={"inputs/datas:0": x_test, "inputs/labels:0": y_test})
    print(">> model accuracy: ",validation_accuracy)    
    return 

#==========  ================================  =====================================  ===============================  ====================
#===========  ================================  =======MAIN==========================  ===============================  ====================
#============  ================================  =====================================  ===============================  ====================
#=============  ================================  =====================================  ===============================  ====================

#Variables de simulation
#reward_threshold = 70
number_of_episodes = 50000
max_steps = 100     

#generate training datas from game simulations
train_datas = generate_training_datas(number_of_episodes, max_step=200, reward_threshold=100, last_steps_to_del=30)

#definition of the network model (creation of the graph)
#Model(learning_rate, dropout_rate, inputoutput_size, hidden_layer_sizes)
learning_rate = 0.0001
dropout_rate = 0.2
io_size = [4,2]
hidden_layer_size = [64,128,256,128,64]
my_model = Model(learning_rate,dropout_rate,io_size,hidden_layer_size)
graph = define_model(my_model)

#opening of a session (initialize the graph variables and make it runable)
sess=open_session(graph)

#split training set into inputs and outputs vector
inputs = []
for episode in train_datas:
    for step in episode:
        inputs.append(step[0])

outputs = []
for episode in train_datas:
    for step in episode:
        outputs.append(step[1])

train_set_len = int(len(inputs)*0.8)
test_set_len = len(inputs)-train_set_len

#train datas 
print('training is starting')
loss = training_run(graph,sess,inputs[0:train_set_len],outputs[0:train_set_len],batch_size=200, number_of_training=5000)
print('training finished')

#validation
validation_run(sess,inputs[train_set_len+1:len(inputs)],outputs[train_set_len+1:len(inputs)])

print('playing')
#if you want to render the game, configure render=True
scores = play(number_of_games=100,win_limit=150, render=False)

#==============================================================================

