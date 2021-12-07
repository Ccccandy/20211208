# -*- coding:utf-8 -*-
"""
用户模拟器，对特征的询问采取随机措施
"""

import pickle
import random
from config import args

class User(object):
    def __init__(self, goal_set):
        self.goal_set = goal_set
        self.max_turn = args.max_turn
        self._init()

    def initialize(self, train_mode=1, epoch_index=None):
        """
        初始化，将goal和state补全
        ：inform的slot，即显性
        ：request的疾病为unk
        ：action为request
        """
        # 初始化
        self._init(train_mode=train_mode,epoch_index=epoch_index)

        # 数据的goal值（即y值）
        goal = self.goal["goal"]

        # 初始action是request的疾病未知
        self.state["action"] = "request"
        self.state["request_slots"]["disease"] = args.value_unknown
        self.state["disease_goal"] = self.goal["disease_tag"]

        # 初始inform是显性的slot
        for slot in goal["explicit_inform_slots"].keys():
            if slot not in self.state['inform_slots']:
                self.state["inform_slots"][slot] = goal["explicit_inform_slots"][slot]
        
        # 初始rest应该只是全部隐形的slot
        for slot in goal["implicit_inform_slots"].keys():
            if slot not in self.state["request_slots"].keys():
                self.state["rest_slots"][slot] = "implicit_inform_slots" 
       
        # 汇总一个user_action
        user_action = self._assemble_user_action()

        return user_action

    def _init(self,train_mode=1,epoch_index=None):
        """
        初始化一个state和一个goal
        """
        self.state = {
            "turn":0,
            "action":None,
            "history":{}, 
            "request_slots":{}, 
            "inform_slots":{}, 
            "rest_slots":{},
            "disease_goal":None
        }     
        # train_mode=0，改变数据集进行测试
        if train_mode == 0:
            self.goal_set = pickle.load(file=open(args.test_input_file, "rb"))
      
        # 1 训练
        if train_mode == 1:
            self.goal = random.choice(self.goal_set["train"])
        # 2 验证
        elif train_mode == 2:
            self.goal = self.goal_set["dev"][epoch_index]
        # 0 测试
        else:
            self.goal = self.goal_set["test"][epoch_index]
            
        # 结束的state、done
        self.episode_over = False
        self.dialogue_status = args.dialogue_status_not_yet

    def _assemble_user_action(self):
        """
        汇总一个user_action
        """
        user_action = {
            "turn":self.state["turn"],
            "action":self.state["action"],
            "speaker":"user",
            "request_slots":self.state["request_slots"],
            "inform_slots":self.state["inform_slots"],
            "disease_goal":self.state["disease_goal"]
        }
        return user_action

    def step(self, agent_action, turn):
        """
        user的step函数，agent_action相当于a
        """
        agent_act_type = agent_action["action"]
        self.state["turn"] = turn

        # 判断是否done
        if self.state["turn"] ==(self.max_turn-2) and agent_act_type == "request":
            # 结束的action、state和done
            self.episode_over = True
            self.state["action"] = "closing" 
            self.dialogue_status = args.dialogue_status_failed
        else:
            pass
       
        if self.episode_over is not True:
            # 把每次已知的slot放入history，并清空
            self.state["history"].update(self.state["inform_slots"])
            self.state["inform_slots"].clear()

            # 对agent的action进行state的变化，并进行user的action的反馈
            if agent_act_type == "inform":
                self._response_inform(agent_action=agent_action)
            elif agent_act_type == "request":
                self._response_request(agent_action=agent_action)

            user_action = self._assemble_user_action()
            reward = self._reward_function()
            return user_action, reward, self.episode_over, self.dialogue_status
        else:
            user_action = self._assemble_user_action()
            reward = self._reward_function()
            return user_action, reward, self.episode_over, self.dialogue_status

    def _response_request(self, agent_action):
        """
        若agent是对症状的判断(request)，user改inform的症状
        """
        for slot in agent_action["request_slots"].keys():
            # 已经问过的slot
            if slot in self.state["history"].keys():     
                # 结束的action、state和done          
                #self.episode_over = True
                #self.state["action"] = "closing" 
                #self.dialogue_status = args.dialogue_status_failed
                self.episode_over = False
                self.state["action"] = "inform" 
                self.dialogue_status = 999
            # 隐性slot
            elif slot in self.goal["goal"]["implicit_inform_slots"].keys():
                # 结束的action、state
                self.state["action"] = "inform"
                self.dialogue_status = args.dialogue_status_right_symptom
                
                self.state["inform_slots"][slot] = self.goal["goal"]["implicit_inform_slots"][slot] # inform中的slot可增加
                if slot in self.state["rest_slots"].keys():     # 删去rest
                    self.state["rest_slots"].pop(slot)
            # 不在goal里的slot
            else:
                self.state["action"] = "inform"
                self.state["inform_slots"][slot] = '3'
  

    def _response_inform(self, agent_action):
        """
        若agent是对疾病的判断(inform)，user改request的疾病
        """
        if "disease" in agent_action["inform_slots"].keys() and agent_action["inform_slots"]["disease"] == self.goal["disease_tag"]:
            # 结束的action、state和done
            self.episode_over = True
            self.state["action"] = "closing" 
            self.dialogue_status = args.dialogue_status_success
               
            self.state["inform_slots"].clear()
            self.state["request_slots"].pop("disease")  # request中的disease可删去
            self.state["history"]["disease"] = agent_action["inform_slots"]["disease"]  # 更新history中的disease

        elif "disease" in agent_action["inform_slots"].keys() and agent_action["inform_slots"]["disease"] != self.goal["disease_tag"]:
            # 结束的action、state和done
            self.episode_over = True
            self.state["action"] = "closing" 
            self.dialogue_status = args.dialogue_status_wrong_disease
            
            self.state["inform_slots"].clear()

    def _reward_function(self):
        """
        reward函数
        """
        if self.dialogue_status == args.dialogue_status_not_yet:
            return args.reward_for_not_yet

        elif self.dialogue_status == args.dialogue_status_success:
            success_reward = args.reward_for_success
            if args.minus_left_slots == 1:
                return success_reward - len(self.state["rest_slots"]) # 
            else:
                return success_reward

        elif self.dialogue_status == args.dialogue_status_failed:
            return args.reward_for_fail

        elif self.dialogue_status == args.dialogue_status_wrong_disease:
            return args.reward_for_wrong_disease
        
        elif self.dialogue_status == args.dialogue_status_right_symptom:
            return args.reward_for_right_symptom
        
        elif self.dialogue_status == 999:
            return -2
           

    def get_goal(self):
        return self.goal

