# -*- coding:utf-8 -*-
"""
对话系统的状态跟踪器，在交互过程中跟踪对话的状态(state)
include：
1、state的初始化
2、state的更新
"""

import copy

class StateTracker(object):
    def __init__(self, user, agent):
        self.user = user
        self.agent = agent
        self._init()

    def initialize(self):
        self._init()

    def _init(self):
        """
        初始化
        """
        self.turn = 0
        self.state = {
            "agent_action":None,
            "user_action":None,
            "turn":self.turn,
            "current_slots":{
                "user_request_slots":{},
                "agent_request_slots":{},
                "inform_slots":{},
                "proposed_disease":{}
            },
            "history":[],
            "disease_goal":None
        }   

    def state_updater(self, user_action=None, agent_action=None):
        """
        state的更新
        ：_state_update_with_user_acion
        ：_state_update_with_agent_action
        """
        assert (user_action is None or agent_action is None), "user action and agent action cannot be None at the same time."
        self.state["turn"] = self.turn
        if user_action is not None:
            self._state_update_with_user_acion(user_action=user_action)
        elif agent_action is not None:
            self._state_update_with_agent_action(agent_action=agent_action)
        self.turn += 1

    def _state_update_with_user_acion(self, user_action):
        """
        通过user action更新state
        """
        # 更新user action
        self.state["user_action"] = user_action
        self.state["disease_goal"] = user_action["disease_goal"]

        # 更新history.append = {"turn":,"action":,"speaker":,"request_slots":,"inform_slots":,'current_slots'}
        temp_action = copy.deepcopy(user_action)
        temp_action["current_slots"] = copy.deepcopy(self.state["current_slots"])   
        self.state["history"].append(temp_action)

        # 获取用户的request疾病
        for slot in user_action["request_slots"].keys():
            self.state["current_slots"]["user_request_slots"][slot] = user_action["request_slots"][slot]

        # 获取用户的inform的slot
        inform_slots = list(user_action["inform_slots"].keys())
        for slot in inform_slots:
            self.state["current_slots"]['inform_slots'][slot] = user_action["inform_slots"][slot]
            if slot in self.state["current_slots"]["agent_request_slots"].keys():
                self.state["current_slots"]["agent_request_slots"].pop(slot)    

    def _state_update_with_agent_action(self, agent_action):
        """
        通过agent action更新state
        """
        # 获取goal中所有显隐性slot
        explicit_implicit_slot_value = copy.deepcopy(self.user.goal["goal"]["explicit_inform_slots"])
        explicit_implicit_slot_value.update(self.user.goal["goal"]["implicit_inform_slots"])

        # 更新agent action
        self.state["agent_action"] = agent_action

        # 更新history.append = {"turn":,"action":,"speaker":,"request_slots":,"inform_slots":,'current_slots'}
        temp_action = copy.deepcopy(agent_action)
        temp_action["current_slots"] = copy.deepcopy(self.state["current_slots"])
        self.state["history"].append(temp_action)

        # 获取agent的request的slot
        for slot in agent_action["request_slots"].keys():
            self.state["current_slots"]["agent_request_slots"][slot] = agent_action["request_slots"][slot]

        # 获取agent的inform的疾病
        for slot in agent_action["inform_slots"].keys():
            slot_value = agent_action["inform_slots"][slot]
            if slot == 'disease' and slot_value == self.user.goal["disease_tag"]:
                self.state["current_slots"]["proposed_disease"][slot] = agent_action["inform_slots"][slot]
            if slot in self.state["current_slots"]["user_request_slots"].keys():
                self.state["current_slots"]["user_request_slots"].pop(slot)  

    def get_state(self):
        return copy.deepcopy(self.state)

    def set_agent(self, agent):
        self.agent = agent
