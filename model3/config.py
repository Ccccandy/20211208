# -*- coding:utf-8 -*-
"""
参数配置

# argparse 是 Python 内置的一个用于命令项选项与参数解析的模块
# 通过在程序中定义好我们需要的参数，argparse 将会从 sys.argv 中解析出这些参数，并自动生成帮助和使用信息。
# 创建 ArgumentParser() 对象
# 调用 add_argument() 方法添加参数 
# "--disease_number"：选项字符串的名字或者列表
# dest：解析后的参数名称
# default：默认值
# 使用 parse_args() 解析添加的参数
# 可以直接args.disease_number，获取disease_number的值
"""

import argparse
import sys
import os
# 手动chdir到指定目录
os.chdir(sys.path[0])

# 初始化参数
disease_number = 6
max_turn = 22

parser = argparse.ArgumentParser()

# TODO: 基本训练参数
# 基本参数
parser.add_argument("--disease_number", dest="disease_number", type=int,default=disease_number,
                    help="疾病种类")
parser.add_argument("--max_turn", dest="max_turn", type=int, default=max_turn, 
                    help="每一回合最大轮回数")
parser.add_argument("--device_for_tf", dest="device_for_tf", type=str, default="/device:CPU:1", 
                    help="tensorflow使用的设备")
# 训练参数
parser.add_argument("--total_epoch_number", dest="total_epoch_number", type=int, default=1000, 
                    help="总的回合数，即训练的回合数")
parser.add_argument("--epoch_size", dest="epoch_size", type=int, default=30, 
                    help="每个epoch包含的一个epi的次数")
parser.add_argument("--evaluate_epoch_number", dest="evaluate_epoch_number", type=int, default=10, 
                    help="验证时，总的回合数，即验证的回合数")
parser.add_argument("--replay_pool_size", dest="replay_pool_size", type=int, default=20000, 
                    help="经验回放池的大小")
parser.add_argument("--warm_start", dest="warm_start",type=int, default=0, 
                    help="是否进行热启动填充经验回放池, {1:Yes; 0:No}")
parser.add_argument("--warm_start_epoch_number", dest="warm_start_epoch_number", type=int, default=50, 
                    help="热启动的回合数")
# 模型内部参数，如batch、input、hidden、output、epsilon、gamma、lr等
parser.add_argument("--batch_size", dest="batch_size", type=int, default=64, 
                    help="训练时的批量大小")
parser.add_argument("--input_size", dest="input_size", type=int, default=369, 
                    help="模型输入层大小")
parser.add_argument("--hidden_size", dest="hidden_size", type=int, default=300, 
                    help="模型隐藏层大小")
parser.add_argument("--epsilon", dest="epsilon", type=float, default=0.1, 
                    help="用于寻找action时，e-greedy贪婪策略的系数")
parser.add_argument("--gamma", dest="gamma", type=float, default=0.9, 
                    help="即时奖励的折扣因子")
parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=0.01, 
                    help="模型的学习率")
# 模型的选择，包括run、agent、model、是否训练
parser.add_argument("--run_id", dest='run_id', type=int, default=1, 
                    help='运行id')
parser.add_argument("--agent_id", dest="agent_id", type=int, default=0, 
                    help="代理id, {0:AgentRule, 1:AgentDQN}")
parser.add_argument("--train_mode", dest="train_mode", type=int, default=1, 
                    help="训练模式, {0:测试, 1:训练, 2:验证}")

# TODO: 模型保存与输出  
# 训练             
parser.add_argument("--save_model", dest="save_model", type=int, default=0,
                    help="是否保存模型, {1:Yes, 0:No}")
parser.add_argument("--checkpoint_path",dest="checkpoint_path", type=str, default="./model/", 
                    help="保存模型断点的目录")
# 验证/测试
parser.add_argument("--model_dir", dest="model_dir", type=str, default="./model/d6_a1_T22/m_d6_a1_s0.515_r14.03_t16.463_wd_e377.ckpt-377",
                    help="保存模型的路径，用于判断是否有可用模型")
# 日志
parser.add_argument("--log_dir", dest="log_dir", type=str, default="./log/", 
                    help="储存训练log的路径")
parser.add_argument("--save_dialogue", dest="save_dialogue", type=int, default=1,
                    help="是否保存对话, {1:Yes, 0:No}")
parser.add_argument("--dialogue_file", dest="dialogue_file", type=str,default="./data/dialogue_output/dialogue_file.txt",
                    help="保存对话内容的路径")
parser.add_argument("--print_result", dest="print_result", type=int, default=1,
                    help="是否打印结果, {1:Yes, 0:No}")

# TODO: data数据集来源
# 训练
parser.add_argument("--goal_set", dest="goal_set", type=str, default='./data/goal_set.p',
                    help='初始的文件路径(包括训练集和验证集)')
parser.add_argument("--slot_set", dest="slot_set", type=str, default='./data/slot_set.p',
                    help='slot集的文件路径')
parser.add_argument("--disease_set", dest="disease_set", type=str,default="./data/disease_set.p",
                    help="疾病文件路径")
parser.add_argument("--disease_symptom", dest="disease_symptom", type=str,default="./data/disease_symptom_set.p",
                    help="疾病及其症状集的文件路径")
# 测试
parser.add_argument("--test_input_file", dest="test_input_file", type=str, default='./data/goal_set_simul.p',
                    help="测试集的文件路径")
parser.add_argument("--test_output_file", dest="test_output_file", type=str, default="./result/result.json", 
                    help="测试集输出的文件路径")


# TODO: reward奖励参数和对应对话状态
# reward奖励参数（需要进行大修改）
parser.add_argument("--reward_for_not_yet", dest="reward_for_not_yet", type=float,default=-1,
                    help="对话还没有结束,这一步没有获取有意义的价值")
parser.add_argument("--reward_for_right_symptom", dest="reward_for_right_symptom", type=float,default=1,
                    help="找到了正确的症状")
parser.add_argument("--reward_for_success", dest="reward_for_success", type=float,default=44,
                    help="对话成功,找到了最后的疾病")
parser.add_argument("--reward_for_fail", dest="reward_for_fail", type=float,default=-22,
                    help="对话失败")
parser.add_argument("--reward_for_wrong_disease", dest="reward_for_wrong_disease", type=float,default=-22,
                    help="找到了错误的疾病")
parser.add_argument("--minus_left_slots", dest="minus_left_slots", type=int, default=0,
                    help="对话成功后是否删去未发现的症状, {1:Yes, 0:No}")
               
# reward对应的对话状态
parser.add_argument("--dialogue_status_failed", dest="dialogue_status_failed", type=int,default=0,
                    help="对话自然结束；同一个slot询问")
parser.add_argument("--dialogue_status_right_symptom", dest="dialogue_status_right_symptom", type=int,default=3,
                    help="找到一个正确的症状")
parser.add_argument("--dialogue_status_success", dest="dialogue_status_success", type=int,default=1,
                    help="找到正确的疾病")                    
parser.add_argument("--dialogue_status_not_yet", dest="dialogue_status_not_yet", type=int,default=-1,
                    help="对话还没结束，无效症状或者初始")
parser.add_argument("--dialogue_status_wrong_disease", dest="dialogue_status_wrong_disease", type=int,default=2,
                    help="找到错误的疾病")
                    
parser.add_argument("--value_unknown", dest="value_unknown", type=str,default="UNK",
                    help="初始化slot和疾病，表示未知")
parser.add_argument("--acc_threshold", dest="acc_threshold", type=float,default=0.3,
                    help="acc阈值")

args = parser.parse_args()

     
