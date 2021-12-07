# -*-coding: utf-8 -*-
"""
最后的训练文件
"""
import copy
import pickle
import json


from agent import AgentRule
from agent import AgentDQN
from user import User
from dialogue_manager import DialogueManager
from config import args


class Train(object):
    """
    训练模型
    """
    def __init__(self,checkpoint_path):       

        self.checkpoint_path = checkpoint_path

        self.epoch_size = args.epoch_size
        self.slot_set = pickle.load(file=open(args.slot_set, "rb"))
        self.goal_set = pickle.load(file=open(args.goal_set, "rb"))
        self.disease_set = pickle.load(file=open(args.disease_set, "rb"))
        self.goal_test_set = pickle.load(file=open(args.test_input_file, "rb"))
        self.disease_symptom = pickle.load(file=open(args.disease_symptom, "rb"))
            
        user = User(goal_set=self.goal_set)
        agent = AgentRule(slot_set=self.slot_set,disease_set=self.disease_set, disease_symptom=self.disease_symptom)
        self.dialogue_manager = DialogueManager(user=user, agent=agent)

        self.best_result = {"success_rate":0.0, "average_reward": 0.0, "average_turn": 0}

    def train_total_epoch(self, agent, total_epoch_number, train_mode=0):
        """
        训练所有的epoch
        """
        save_model = args.save_model

        self.dialogue_manager.set_agent(agent=agent)

        print('开始训练')
        for index in range(total_epoch_number):
            print('===第',index,'次epoch开始===')

            # 训练dqn
            if train_mode == 1:
                if isinstance(self.dialogue_manager.state_tracker.agent, AgentDQN):
                    # 若经验池不够，则不会进行训练
                    self.dialogue_manager.train()
                    # 将一个epoch的数据放入经验池中
                    self.get_one_epoch(epoch_size=self.epoch_size,train_mode=train_mode)
                


            # 边训练边验证进行保存
            test_mode = 2
            result = self.evaluate(agent,test_mode)

            # 非训练不进行保存
            if train_mode != 1:
                break
            
            if result["success_rate"] >= self.best_result["success_rate"] and result["success_rate"] > args.acc_threshold  and train_mode==1:
                if save_model == 1 and train_mode == 1:
                    self.dialogue_manager.state_tracker.agent.save_model(model_performance=result, episodes_index = index, checkpoint_path=self.checkpoint_path)
                    print("模型更新，且保存")
                else:
                    pass
                self.best_result = copy.deepcopy(result)
            else:
                if save_model == 1 and train_mode == 1:
                    result['success_rate'] = 0
                    result['average_reward'] = 0
                    result['average_turn'] = 0
                    self.dialogue_manager.state_tracker.agent.save_model(model_performance=result, episodes_index = total_epoch_number, checkpoint_path=self.checkpoint_path)
                    print("最优模型已存在，不进行保存")   

    def evaluate(self, agent, test_mode, print_result = 0):
        """
        检验 or 测试
        train_mode不为1, greedy_strategy为0
        """
        self.dialogue_manager.set_agent(agent=agent)

        success_count = 0
        total_reward = 0
        total_turn = 0
        recall_score = 0

        # 检验     
        if test_mode == 2:
            goal_set = self.goal_set["dev"]
            evaluate_epoch_number = len(goal_set)
        # 测试
        else:
            goal_set = self.goal_test_set['test']
            evaluate_epoch_number = len(goal_set)

        if print_result:
            pre_set = self.json_bulid(goal_set)
        #a = 0   
        #print('test number:',evaluate_epoch_number)
        # 开始预测                            
        for epoch_index in range(evaluate_epoch_number):
            self.dialogue_manager.initialize(train_mode=test_mode, epoch_index=epoch_index)
            episode_over = False
            ids = goal_set[epoch_index]['consult_id']
            
            while episode_over == False:
                reward, episode_over, dialogue_status, agent_action = self.dialogue_manager.step(save_record=False,train_mode=test_mode,greedy_strategy=0)
                
                total_reward += reward
                # 如果有,获取disease
                if print_result and agent_action['inform_slots']: 
                    pre_set[ids]['Disease'] = list(agent_action['inform_slots'].values())[0]                       
            
            #if self.dialogue_manager.state_tracker.turn==21:
            #    a = a+1
            #print(a)
            total_turn += self.dialogue_manager.state_tracker.turn

            # slot召回率
            predict_state = self.dialogue_manager.state_tracker.state['current_slots']['inform_slots']
            goal_state = set(list(self.dialogue_manager.state_tracker.user.goal['goal']['explicit_inform_slots'].keys()) + list(self.dialogue_manager.state_tracker.user.goal['goal']['implicit_inform_slots'].keys()))            
            correct_slot = 0
            for key in predict_state.keys():
                if key in goal_state:
                    correct_slot += 1
            recall_score += correct_slot / len(goal_state)
            # 疾病准确率
            if dialogue_status == args.dialogue_status_success:
                success_count += 1

            if print_result:
                pre_set[ids]['Symptoms'] = list(predict_state.keys())

        if print_result:
            with open(args.test_output_file,'w') as f:
                json.dump(pre_set, f)
            
        average_recall = float("%.3f" % (float(recall_score) / evaluate_epoch_number))
        success_rate = float("%.3f" % (float(success_count) / evaluate_epoch_number))
        average_reward = float("%.3f" % (float(total_reward) / evaluate_epoch_number))
        average_turn = float("%.3f" % (float(total_turn) / evaluate_epoch_number))
        res = {"success_rate":success_rate, "average_reward": average_reward, "average_turn": average_turn, 'average_recall': average_recall}

        print("验证or测试的结果： SR %s, ave reward %s, ave turns %s,  ave recall %s" % (res['success_rate'], res['average_reward'], res['average_turn'], res["average_recall"]))
        
        return res

    def get_one_epoch(self, epoch_size,train_mode):
            """
            获取epoch_size个epoch的数据，将其放入经验池
            ：可以是dqn获取
            ：可以是rule获取
            """
            success_count = 0
            total_reward = 0
            total_turns = 0
            for i in range(epoch_size):
                self.dialogue_manager.initialize(train_mode=args.train_mode)
                episode_over = False
                while episode_over == False:
                    reward, episode_over, dialogue_status, agent_action = self.dialogue_manager.step(save_record=True,train_mode=train_mode,greedy_strategy=1)
                    total_reward += reward
                total_turns += self.dialogue_manager.state_tracker.turn
                if dialogue_status == args.dialogue_status_success:
                    success_count += 1
            success_rate = float("%.3f" % (float(success_count) / epoch_size))
            average_reward = float("%.3f" % (float(total_reward) / epoch_size))
            average_turn = float("%.3f" % (float(total_turns) / epoch_size))
            res = {"success_rate":success_rate, "average_reward": average_reward, "average_turn": average_turn}
           
            return res

    def warm_start(self, agent, warm_start_epoch_number):
        """
        热启动对话，使用基于规则的代理的示例填充DQN的经验池
        """
        self.dialogue_manager.set_agent(agent=agent)

        for index in range(warm_start_epoch_number):
            print('---第%d个热启动开始---'%(index))
            res = self.get_one_epoch(epoch_size=self.epoch_size,train_mode=1)
            print("第%3d个热启动的  SR %s,ave reward %s, ave turns %s" % (index, res['success_rate'], res['average_reward'], res['average_turn']))
        
    def json_bulid(self, goal_set):
        """
        构建goal_slots文件
        ：{id1：{'Symptoms'：[],'Disease':None}，id2：{}...}
        """
        goal_slots = dict()
        for goal in goal_set:
            ids = goal['consult_id']
            goal_slots[ids] = dict()
            goal_slots[ids]['Symptoms'] = list()
            goal_slots[ids]['Disease'] = None
        return goal_slots
