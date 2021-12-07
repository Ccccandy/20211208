# -*- coding:utf-8 -*-
"""
对话管理器
"""

import random
import numpy as np
from collections import deque
from state_tracker import StateTracker
from agent import AgentDQN
from agent import AgentRule
from config import args

class DialogueManager(object):
    """
    对话系统的对话管理器
    """
    def __init__(self, user, agent):
      
        self.state_tracker = StateTracker(user=user, agent=agent)

        self.replay_pool = deque(maxlen=args.replay_pool_size)
        
        self.save_dialogue = args.save_dialogue
        self.dialogue_output_file = args.dialogue_file


    def initialize(self,train_mode=1, epoch_index=None):
        """
        初始化state，再初始化user，通过user更新state，再初始化agent
        ：user提供显性的inform
        ：state获取显性的inform
        ：agent为空
        """
        # 初始化s
        self.state_tracker.initialize()
        # 初始化user和agent
        user_action = self.state_tracker.user.initialize(train_mode = train_mode, epoch_index=epoch_index)
        self.state_tracker.state_updater(user_action=user_action)
        self.state_tracker.agent.initialize()
 
    def step(self,save_record,train_mode, greedy_strategy):
        """
        接下来的两回合对话。代理将首先执行操作，然后执行用户模拟器。
        ：返回reward, episode_over,dialogue_status, agent_action
        ：record_training_sample 将(s,a,s',r,done)放入经验池
        ：__output_dialogue 记录整个对话，state和goal的输出
        """
        # s
        state = self.state_tracker.get_state()
        # a
        
        agent_action, action_index = self.state_tracker.agent.sample_action(state=state,turn=self.state_tracker.turn,greedy_strategy=greedy_strategy)      
        #print(agent_action)
        # s变化
        self.state_tracker.state_updater(agent_action=agent_action)       
        # (a,s)--r、done
        user_action, reward, episode_over, dialogue_status = self.state_tracker.user.step(agent_action=agent_action,turn=self.state_tracker.turn)
        # s变化
        self.state_tracker.state_updater(user_action=user_action)
        

        # 将(s,a,s',r,done,d)放入经验池
        if save_record == True:

            self.record_training_sample(
                state=state,
                agent_action=action_index,  
                next_state=self.state_tracker.get_state(),
                reward=reward,
                episode_over=episode_over
            )
        else:
            pass
        

        # 测试才打印
        if episode_over == True and self.save_dialogue == 1 and train_mode == 0:
            state = self.state_tracker.get_state()
            goal = self.state_tracker.user.get_goal()
            self.__output_dialogue(state=state, goal=goal)

        return reward, episode_over,dialogue_status, agent_action

    def train(self):
        """
        根据不同的agent训练不同的模型
        """
        if isinstance(self.state_tracker.agent, AgentDQN):
            self.__train_dqn()
            # 两个网络需要更新
            self.state_tracker.agent.update_target_network()
 


    def __train_dqn(self):
        """
        训练dqn模型
        ：在replay_pool中随机选择batch个数据，进行train
        ：若不够就不行训练，一直等到够了才进行训练
        """
        total_loss = 0.0
        d_loss = 0.0
        batch_size = args.batch_size
        
        replay_num = len(self.replay_pool)      
        shuffle_index = np.arange(replay_num)
        np.random.shuffle(shuffle_index)

        replay_np = np.array(self.replay_pool)

        for i in range(0,len(self.replay_pool),batch_size):
            end_index = i + batch_size
            batch_index = shuffle_index[i:end_index]
            batch = replay_np[batch_index]
            loss = self.state_tracker.agent.train(batch=batch)
            total_loss += loss["loss"]
            #d_loss += loss["d_loss"]
        #print("dqn一个batch的loss %.4f,d_loss %.4f, replay pool %s" % (float(total_loss) / (len(self.replay_pool)+1e-12),float(d_loss) / (len(self.replay_pool)+1e-12), len(self.replay_pool)))
        print('dqn的loss%.4f,replay pool %s'%(float(total_loss) / (len(self.replay_pool)+1e-12), len(self.replay_pool)))

    def record_training_sample(self, state, agent_action, reward, next_state, episode_over):
        """
        将(s,a,s',r,done)放入经验池
        """
        disease = self.state_tracker.agent.disease_to_rep(state)
        state = self.state_tracker.agent.state_to_rep(state)
        next_state = self.state_tracker.agent.state_to_rep(next_state)
        
        self.replay_pool.append((state, agent_action, reward, next_state, episode_over,disease))
       
    # 输出所有对话的历史记录
    def __output_dialogue(self,state, goal):
        """
        记录整个对话，state和goal的输出
        """
        history = state["history"]
        file = open(file=self.dialogue_output_file,mode="a+",encoding="utf-8")
        file.write("User goal: " + str(goal)+"\n")
        for turn in history:
            speaker = turn["speaker"]
            action = turn["action"]
            inform_slots = turn["inform_slots"]
            request_slots = turn["request_slots"]
            file.write(speaker + ": " + action + "; inform_slots:" + str(inform_slots) + "; request_slots:" + str(request_slots) + "\n")
        file.write("\n\n")
        file.close()

    def set_agent(self,agent):
        """
        获取当下的agent
        """
        self.state_tracker.set_agent(agent=agent)