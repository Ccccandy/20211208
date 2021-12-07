# -*-coding:utf-8 -*
"""
基于dqn的agent
"""
import numpy as np
import random
from agent import Agent
import sys
sys.path.append(sys.path[0].replace("agent",""))
from config import args
from policy_learning import DQN_dis

class AgentDQN(Agent):
    def __init__(self, slot_set, disease_set, disease_symptom):
        super(AgentDQN, self).__init__(slot_set=slot_set,disease_set=disease_set,disease_symptom=disease_symptom)
        
        input_size = args.input_size
        hidden_size = args.hidden_size
        output_size = len(self.action_space)
        self.dis2slot=np.load('./data/dis2slot.npy')
        self.slot2dis=np.load('./data/slot2dis.npy')
        self.p_slot=np.load('./data/p_slot.npy')


        self.dqn = DQN_dis(input_size=input_size, hidden_size=hidden_size,output_size=output_size)

    def sample_action(self, state, turn,greedy_strategy):
        '''
        根据greedy策略获取action
        ：1 则采用e-greedy，随机或最大的action(st;θ)
        ：0 就直接采用最大的action(st;θ)
        '''
        self.agent_action["turn"] = turn

        state_rep = self.state_to_rep(state=state)  # （347+22，
        #print('----') 
        #print(state_rep[365])
        #print('----')
        state = state_rep[:347].reshape(1,-1)
        mask = state!=1
        mask2 = state!=0
        final_p_slot = self.p_slot.copy()
        final_p_slot[mask] = 0        
        final_dis = np.matmul(final_p_slot,self.slot2dis) #（1,6
        final_sym = np.matmul(final_dis,self.dis2slot)  
        final_sym[mask2] = 0
        final_p = np.hstack((final_sym,final_dis))
        #print('kg',np.argmax(final_p[0][:347]),max(final_p[0]))
        model_y=self.dqn.predict(Xs=[state_rep])[0]      # （1，347,应该是每个都21轮 
        #print('model',np.argmax(model_y[0][:347]),max(model_y[0]))
        final = final_p+model_y
        max_index = np.argmax(final[0][:347])
        #print('final',max_index)
        dis = self.dqn.predict_DIS(Xs=[state_rep])[0] #（1,6

        final_dis = final_dis+dis

        if state_rep[365] == 1:
            #print('进去了')
            max_index = np.argmax(final_dis, axis=1)[0]
            max_index = max_index + 347
            #print(max_index)

        
        if greedy_strategy == 1:
            greedy = random.random()
            if greedy < args.epsilon:
                action_index = random.randint(0, len(self.action_space) - 1) # random select a action
            else:
                action_index = max_index
        else:
            action_index = max_index
        
        #print('turn',turn,'insex',action_index)
        agent_action = self.action_space[action_index]
        agent_action["turn"] = turn
        agent_action["speaker"] = "agent"
        return agent_action, action_index

    def train(self, batch):
        '''
        优化一个batch的数据
        '''
        loss = self.dqn.singleBatch(batch=batch)
        return loss

    def update_target_network(self):
        '''
        更新目标网络
        '''
        self.dqn.update_target_network()

    def save_model(self, model_performance,episodes_index, checkpoint_path = None):
        '''
        保存模型
        '''
        self.dqn.save_model(model_performance=model_performance, episodes_index = episodes_index, checkpoint_path=checkpoint_path)
