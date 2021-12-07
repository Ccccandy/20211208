# -*- coding:utf-8 -*-
"""
主程序
"""
import pickle
from agent import AgentDQN
from agent import AgentRule
import random
from train import Train
from config import args

def main():
    """
    train_mode 1:训练 0：测试 2：验证
    """
    disease_number = args.disease_number
    max_turn = args.max_turn
    agent_id = args.agent_id  
    checkpoint_path = "./model/d" + str(disease_number) + "_a" + str(agent_id)  + "_T" + str(max_turn) +  "/"
    
    warm_start = args.warm_start
    warm_start_epoch_number = args.warm_start_epoch_number
    total_epoch_number = args.total_epoch_number

    print_result = args.print_result

    slot_set = pickle.load(file=open(args.slot_set, "rb"))
    disease_set = pickle.load(file=open(args.disease_set, "rb"))
    disease_symptom = pickle.load(file=open(args.disease_symptom, "rb"))

    train_mode = args.train_mode   
    train_model = Train(checkpoint_path=checkpoint_path)
      
    # 热启动
    if warm_start == 1 and train_mode == 1:
        print("===开始热启动===")
        agent = AgentRule(slot_set=slot_set, disease_set=disease_set,disease_symptom=disease_symptom)
        train_model.warm_start(agent=agent,warm_start_epoch_number=warm_start_epoch_number)

    # agent，即模型
    if agent_id == 0:
        agent = AgentRule(slot_set=slot_set, disease_set=disease_set,disease_symptom=disease_symptom)
    elif agent_id == 1:
        agent = AgentDQN(slot_set=slot_set, disease_set=disease_set,disease_symptom=disease_symptom)


    # 训练还是测试
    if train_mode == 1:
        train_model.train_total_epoch(agent=agent,total_epoch_number=total_epoch_number, train_mode=train_mode)
    else:        
        train_model.evaluate(agent=agent, test_mode = train_mode, print_result = print_result)

if __name__ == "__main__":
    random.seed(5)
    main()