# -*- coding: utf-8 -*-
"""
基于规则的agent
:rule p(a) = yes_score - 0.5*not_sure_score - deny_score
"""
from agent import Agent
import copy
import numpy as np
import random
import sys
sys.path.append(sys.path[0].replace("agent",""))
from config import args


class AgentRule(Agent):
    """
    基于规则的agent
    """
    def __init__(self,slot_set, disease_set, disease_symptom):
        super(AgentRule,self).__init__(slot_set=slot_set,disease_set=disease_set,disease_symptom=disease_symptom)

    def sample_action(self, state, turn, greedy_strategy):
        '''
        根据规则获取最大概率的action
        ：无候选，inform疾病
        ：有候选，request slot
        '''
        candidate_disease_symptoms = self._get_candidate_disease_symptoms(state=state)

        match_disease = candidate_disease_symptoms["disease"]
        candidate_symptoms = candidate_disease_symptoms["candidate_symptoms"]
        
        self.agent_action["request_slots"].clear()
        self.agent_action["inform_slots"].clear()
        self.agent_action["turn"] = turn

        if len(candidate_symptoms) == 0:
            self.agent_action["action"] = "inform"
            self.agent_action["inform_slots"]["disease"] = match_disease
        else:
            symptom = random.choice(candidate_symptoms)
            self.agent_action["action"] = "request"
            self.agent_action["request_slots"][symptom] = args.value_unknown

        agent_action = copy.deepcopy(self.agent_action)
        agent_action.pop("turn")
        agent_action.pop("speaker")
        agent_index = self.action_space.index(agent_action)


        return self.agent_action, agent_index

    
    def _get_candidate_disease_symptoms(self, state):
        """
        将现有slot与疾病症状进行compare，判断用户可能有什么疾病
        ：最有可能的疾病
        ：候选疾病slot
        """
        inform_slots = state["current_slots"]["inform_slots"]

        # 初始化全部的疾病，{'D1':{'yes':0,'not_sure':0,'deny':,0},'D2':{..}...}
        disease_match_number = {}      
        for disease in self.disease_symptom.keys():
            disease_match_number[disease] = {}
            disease_match_number[disease]["yes"] = 0
            disease_match_number[disease]["not_sure"] = 0
            disease_match_number[disease]["deny"] = 0

        # 遍历已知的疾病
        for slot in inform_slots.keys():
            for disease in disease_match_number.keys():
                if slot in self.disease_symptom[disease]["symptom"] and inform_slots[slot] == '1':
                    disease_match_number[disease]["yes"] += 1
                elif slot in self.disease_symptom[disease]["symptom"] and inform_slots[slot] == '2':
                    disease_match_number[disease]["not_sure"] += 1
                elif slot in self.disease_symptom[disease]["symptom"] and inform_slots[slot] == '0':
                    disease_match_number[disease]["deny"] += 1

        # 计算疾病的概率
        disease_score = {}
        for disease in disease_match_number.keys():
            yes_score = float(disease_match_number[disease]["yes"]) / len(self.disease_symptom[disease]["symptom"])
            not_sure_score = float(disease_match_number[disease]["not_sure"]) / len(self.disease_symptom[disease]["symptom"])
            deny_score = float(disease_match_number[disease]["deny"]) / len(self.disease_symptom[disease]["symptom"])
            disease_score[disease] = yes_score - 0.5*not_sure_score - deny_score

        # 获取更有可能的疾病（并列只取第一个）
        match_disease = max(disease_score.items(), key=lambda x: x[1])[0] 

        # 获取该疾病的候选特征
        candidate_symptoms = []
        for symptom in self.disease_symptom[match_disease]["symptom"]:
            if symptom not in inform_slots.keys():
                candidate_symptoms.append(symptom)
        return {"disease":match_disease,"candidate_symptoms":candidate_symptoms}