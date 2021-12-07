# -*- coding: utf-8 -*-
"""
agent基类
"""

import numpy as np
import copy
import sys
sys.path.append(sys.path[0].replace("agent",""))
from config import args


class Agent(object):
    """
    agent基类
    """
    def __init__(self, slot_set, disease_set, disease_symptom):
        self.slot_set = slot_set
        self.disease_set = disease_set
        self.disease_symptom = self.disease_symptom_clip(disease_symptom)
        self.candidate_disease_list = []
        self.candidate_symptom_list = []
        self.action_space = self._build_action_space()
        self.agent_action = {
            "turn":1,
            "action":None,
            "speaker":"agent",
            "request_slots":{},
            "inform_slots":{}          
        }

    def initialize(self):
        """
        初始化
        """
        self.candidate_disease_list = []
        self.candidate_symptom_list = []
        self.agent_action = {
            "turn":None,
            "action":None,
            "speaker":"agent",
            "request_slots":{},
            "inform_slots":{}           
        }

    def sample_action(self, state, turn, greedy_strategy):
        """
        基于不同的方法采取行动
        ：rule
        ：dqn
        """
        return self.agent_action

    def train(self, batch):
        """
        批量训练模型
        """
        pass

    def state_to_rep(self, state):
        """
        将对话状态（slot、turn）映射到向量    
        """
        current_slots = copy.deepcopy(state["current_slots"]["inform_slots"])
        current_slots_rep = np.zeros(len(self.slot_set.keys()))
        for slot in current_slots.keys():
            if current_slots[slot] == '1':
                if slot in self.slot_set.keys():
                    current_slots_rep[self.slot_set[slot]] = 1.0              
            elif current_slots[slot] == '0':
                if slot in self.slot_set.keys():
                    current_slots_rep[self.slot_set[slot]] = -1.0             
            elif current_slots[slot] == '2':
                if slot in self.slot_set.keys():
                    current_slots_rep[self.slot_set[slot]] = -2.0
        turn_rep = np.zeros(args.max_turn)
        turn_rep[state["turn"]] = 1.0
        # shape =(Len(slot_set)+max_turn,1)
        state_rep = np.hstack((current_slots_rep, turn_rep))
        return state_rep

    def disease_to_rep(self, state):
        """
        将疾病映射到向量    
        """
        disease = state["disease_goal"]
        disease_rep = np.zeros(len(self.disease_set.keys()))
        if disease in self.disease_set.keys():
            disease_index = self.disease_set[disease]
            disease_rep[disease_index] = 1.0
        return disease_rep
 

    def _build_action_space(self):
        """
        构建行为空间
        ：requet：slot-unk
        ：inform：disease-disease
        """
        action_spaces = []
        for slot in self.slot_set.keys():
            action_spaces.append({'action': 'request', 'inform_slots': {}, 'request_slots': {slot: args.value_unknown}})
        for disease in self.disease_symptom.keys():
            action_spaces.append({'action': 'inform', 'inform_slots': {"disease":disease}, 'request_slots': {}})

        return action_spaces

    def disease_symptom_clip(self, disease_symptom):
        """
        疾病症状切片，最多每个疾病只取22/2.5个疾病症状
        ：{{'index': 5, 'Symptom': {'喷嚏': 4},'symptom':{'喷嚏': 4},...}
        """
        max_turn = args.max_turn
        temp_disease_symptom = copy.deepcopy(disease_symptom)

        for key, value in disease_symptom.items():
            symptom_list = sorted(value['Symptom'].items(),key = lambda x:x[1],reverse = True)
            symptom_list = [v[0] for v in symptom_list]
            symptom_list = symptom_list[0:min(len(symptom_list), int(max_turn / 3))]
            temp_disease_symptom[key]['symptom'] = symptom_list

        return temp_disease_symptom