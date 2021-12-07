# -*- coding:utf-8 -*-
"""
dqn：基于值的RL
:目标函数 loss(θ) = 1/m[r(t) + λmaxQ(s(t+1);θ') - Q(s(t);θ)]² + reg(θ)
"""

import numpy as np
import tensorflow as tf
import os
import sys
sys.path.append(sys.path[0].replace("policy_learning",""))
from config import args

os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.chdir(sys.path[0].replace("policy_learning",""))

class DQN_dis(object):
    """
    dqn：只用了一层hidden layer
    """
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size#不输出疾病
        self.disease_size = args.disease_number
        self.checkpoint_path = args.checkpoint_path
        self.log_dir = args.log_dir        
        self.learning_rate = args.learning_rate
        self.update_target_network_operations = []
        self.__build_model()

    def __build_model(self):
        """
        构建基础模型
        """
        # 构建图
        device = args.device_for_tf
        with tf.device(device):
            self.graph = tf.Graph()
            with self.graph.as_default():

                self.input = tf.placeholder(dtype=tf.float64, shape=(None, self.input_size), name="input")
                self.target_value = tf.placeholder(dtype=tf.float64, shape=(None, self.output_size), name="target_value")
                self.target_disease = tf.placeholder(dtype=tf.float64, shape=(None, self.disease_size), name="target_disease")                
                
                # 疾病网络
                with tf.variable_scope(name_or_scope="disease_network"):
                    self.disease_network_variables = {}
                    self.disease_output = self._build_disease_layer(variables_dict=self.disease_network_variables,
                                                       input=self.input,
                                                       input_size=self.input_size,
                                                       output_size=self.disease_size,
                                                       weights_key="w1",
                                                       bias_key="b1"
                                                       )
                # 目标网络
                with tf.variable_scope(name_or_scope="target_network"):
                    self.target_network_variables = {}
                    self.target_output = self._build_layer(variables_dict=self.target_network_variables,
                                                       input=self.input,
                                                       input_size=self.input_size,
                                                       output_size=self.output_size,
                                                       weights_key="w1",
                                                       bias_key="b1")
                # 当前网络
                with tf.variable_scope(name_or_scope="current_network"):
                    self.current_network_variables = {}
                    self.current_output = self._build_layer(variables_dict=self.current_network_variables,
                                                       input=self.input,
                                                       input_size=self.input_size,
                                                       output_size=self.output_size,
                                                       weights_key="w1",
                                                       bias_key="b1")
                    for key, value in self.current_network_variables.items():
                        if "w" in key: 
                            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, value)
                
                # 当前网络更新目标网络
                with tf.name_scope(name="ops_of_updating_target_network"):
                    self.update_target_network_operations = self._update_target_network_operations()

                self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
                reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                self.reg_loss = tf.contrib.layers.apply_regularization(self.regularizer, reg_variables)
                self.v_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.target_value - self.current_output),axis=1),name="loss") 
                self.loss = self.v_loss+ self.reg_loss

                tf.summary.scalar("loss", self.loss)
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss= self.loss)
                
                self.d_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.target_disease - self.disease_output),axis=1),name="d_loss") 
                tf.summary.scalar("d_loss", self.d_loss)
                self.d_optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(loss= self.d_loss)
                
                
                
                self.initializer = tf.global_variables_initializer()
                self.model_saver = tf.train.Saver()
            self.graph.finalize()

        # 可视化log
        self.summary_writer = tf.summary.FileWriter(logdir=self.log_dir + "train", graph=self.graph)

        # tf的创建初始化、gpu的设定
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 占用GPU90%的显存 
        config.log_device_placement = False
        self.session = tf.Session(graph=self.graph,config=config)
        self.session.run(self.initializer)

        # 非1 不训练
        if args.train_mode != 1:
            self.restore_model(args.model_dir)


    def singleBatch(self, batch):
        """
        一个bacth
        ：state, agent_action, reward, next_state, episode_over
        ：loss = (b,)
        """
        Xs = []
        next_Xs = []
        Ds = []
        for i,x in enumerate(batch):
            state_pre = x[0]
            next_state_pre = x[3]
            disease_pre = x[5]
            Xs.append(state_pre)
            next_Xs.append(next_state_pre)
            Ds.append(disease_pre)


        next_Ys = self._predict_target(Xs=next_Xs)[0]
        Ys_label = self.predict(Xs=Xs)[0] 

        for i, x in enumerate(batch):
            reward = x[2] 
            episode_over = x[4] 
            action = x[1] 
            next_max_y = np.max(next_Ys[i])
            target_y = reward
            if not episode_over:
                target_y = float(args.gamma * next_max_y + reward)
            Ys_label[i][action] = target_y
        # target_value就是在现在的a下，获得的Q（st+1，a；θ’）值

        feed_dict = {self.input:Xs, self.target_value:Ys_label, self.target_disease:Ds}
        loss = self.session.run(self.loss,feed_dict=feed_dict) 
        d_loss = self.session.run(self.d_loss,feed_dict=feed_dict) 
        self.session.run((self.optimizer,self.d_optimizer), feed_dict=feed_dict)
        return {"loss": loss,'d_loss':d_loss}
        
    def predict_DIS(self, Xs):
        """
        当前网络θ的预测
        ：根据st预测at
        ：p(at)=(b,a) max(at)=(b,)
        """
        feed_dict = {self.input:Xs}
        dis = self.session.run(self.disease_output,feed_dict=feed_dict)
        max_index = np.argmax(dis, axis=1)
        return dis, max_index[0]


    def predict(self, Xs):
        """
        当前网络θ的预测
        ：根据st预测at
        ：p(at)=(b,a) max(at)=(b,)
        """
        feed_dict = {self.input:Xs}
        Ys = self.session.run(self.current_output,feed_dict=feed_dict)
        max_index = np.argmax(Ys, axis=1)
        return Ys, max_index[0]

    def _predict_target(self, Xs):
        """
        目标网络θ’的预测
        ：根据st预测at
        ：p(at)=(b,a) max(at)=(b,)
        """
        feed_dict ={self.input:Xs}
        Ys = self.session.run(self.target_output,feed_dict=feed_dict)
        max_index = np.argmax(Ys, axis=1)
        return Ys, max_index[0]

    def _build_disease_layer(self, variables_dict, input, input_size, output_size, weights_key, bias_key):
        """
        构建疾病layer
        """
        with self.graph.as_default():

            weights = tf.get_variable(name=weights_key, shape=(input_size, output_size), dtype=tf.float64)
            bias = tf.get_variable(name=bias_key, shape=(output_size), dtype=tf.float64)
            variables_dict[weights_key] = weights
            variables_dict[bias_key] = bias
            tf.summary.scalar(name=weights_key, tensor=weights)
            tf.summary.scalar(name=bias_key,tensor=bias)

            output = tf.nn.softmax(tf.add(tf.matmul(input, weights), bias),name="disease_output")
            #output = tf.nn.sigmoid(tf.add(tf.matmul(input, weights), bias),name="disease_output")
        return output


    def _build_layer(self, variables_dict, input, input_size, output_size, weights_key, bias_key):
        """
        构建一层神经网络
        """
        with self.graph.as_default():

            weights = tf.get_variable(name=weights_key, shape=(input_size, output_size), dtype=tf.float64)
            bias = tf.get_variable(name=bias_key, shape=(output_size), dtype=tf.float64)
            variables_dict[weights_key] = weights
            variables_dict[bias_key] = bias
            tf.summary.scalar(name=weights_key, tensor=weights)
            tf.summary.scalar(name=bias_key,tensor=bias)

            output = tf.nn.relu(tf.add(tf.matmul(input, weights), bias)) #（b，347
            
            output = tf.nn.sigmoid((output),name="output")
        return output

    def update_target_network(self):
        """
        更新网络网络
        ：run下面的update_target_network
        """
        self.session.run(fetches=self.update_target_network_operations)

    def _update_target_network_operations(self):
        """
        更新网络网络（这是一个硬更新，可以使用软更新）
        ：一个assign赋值的过程的列表
        ：需要update_target_network进行run
        """
        update_target_network_operations = []
        for key in self.current_network_variables.keys():
            update = tf.assign(ref=self.target_network_variables[key],value=self.current_network_variables[key].value())
            update_target_network_operations.append(update)
        return update_target_network_operations

    def save_model(self, model_performance,episodes_index, checkpoint_path = None):
        """
        保存模型
        """
        print('保存训练好的模型...')
        if checkpoint_path == None: 
            checkpoint_path = self.checkpoint_path

        agent_id = args.agent_id
        disease_number = args.disease_number
        success_rate = model_performance["success_rate"]
        average_reward = model_performance["average_reward"]
        average_turn = model_performance["average_turn"]
        model_file_name = "m_d" + str(disease_number) + "_a" + str(agent_id) + "_s" + str(success_rate) + "_r" + str(average_reward) + "_t" + str(average_turn) + "_wd" + "_e" + str(episodes_index) + ".ckpt"
        
        save_path = checkpoint_path + model_file_name 

        self.model_saver.save(sess=self.session,save_path=save_path,global_step=episodes_index)

    def restore_model(self, saved_model):
        """
        加载模型
        """
        print("加载训练好的模型...")
        self.model_saver.restore(sess=self.session,save_path=saved_model)