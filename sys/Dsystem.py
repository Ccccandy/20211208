  
from flask import Flask, render_template, request
from test import entity
from test2 import dis
import json
import time

app = Flask(__name__)

a = [2]

with open('data2/sym.json', 'r', encoding='utf-8') as fr:
    sym = json.load(fr)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/get")
def get_bot_response():
    print(len(a))
    
    userText = request.args.get('msg')
    if len(a)==1:
        print('用户第一次输入信息为：',userText)
        turn = a[0]
        syms = entity(userText)
        a.append(syms)
        print("症状识别模型的结果为：",syms)
        resp = dis(syms,turn)
        print('疾病诊断模型的结果为：',resp)
        ##加一个症状和检查报告的分开！！！！
        while resp in syms.keys():
            print('后台error备注：该症状/检查报告已经问过了')
            a[0] += 2
            turn = a[0]
            resp = dis(syms,turn)
            print('模型诊断的是：',resp)
        a.append(resp)
        if resp in ['小儿支气管炎','上呼吸道感染','小儿腹泻','小儿发热','小儿感冒','小儿消化不良']:
            dict_data = {}
            dict_data[str('您好，宝宝可能是得了%s了'%(resp))] = 0
            return dict_data
        elif resp in sym:
            dict_data = {}
            dict_data[str('请问宝宝最近有%s吗？如果有，请回复1；如果没有，请回复2；如果不确定，请回复3'%(resp))] = 0
            return dict_data
        else:
            dict_data = {}
            dict_data[str('请问宝宝最近有做过%s检查吗？如果有，请回复1；如果没有，请回复2；如果不确定，请回复3'%(resp))] = 0
            return dict_data

    elif a[0]==18:
        if "谢" in userText:
            dict_data = {}
            dict_data['非常感谢您对本系统的支持，祝您的宝宝早日恢复健康！'] = 0
            return dict_data  
        else:
            dict_data = {}
            dict_data[str('本次就诊已结束，祝您的宝宝早日恢复健康！')] = 0
            return dict_data
    
    else:
        if userText in ['1','2','3']:
            a[0] += 2
            turn = a[0]
            #print(turn)
            temp = a[-2]
            temp[a[-1]] = int(userText)
            a[-1] = temp
            print("症状识别模型的结果为：",temp)
            resp = dis(temp,turn)
            print('疾病诊断模型的结果为：',resp)
            while resp in temp.keys():
                print('后台error备注：该症状/检查报告已经问过了')
                a[0] += 2
                turn = a[0]
                resp = dis(temp,turn)   
                print('疾病诊断模型的结果为：',resp)
            a.append(resp)
            if resp in ['小儿支气管炎','上呼吸道感染','小儿腹泻','小儿发热','小儿感冒','小儿消化不良']:
                dict_data = {}
                dict_data[str('您好，宝宝可能是得了%s了'%(resp))] = 0
                
                id = int(time.time())
                sy = a[-2]
                di = a[-1]
                total_dic = {
                    '用户id':id,
                    '用户症状':sy,
                    '用户可能疾病':di
                }
                print(total_dic)
                return dict_data
            elif resp in sym:
                dict_data = {}
                dict_data[str('请问宝宝最近有%s吗？如果有，请回复1；如果没有，请回复2；如果不确定，请回复3'%(resp))] = 0
                return dict_data
            else:
                dict_data = {}
                dict_data[str('请问宝宝最近有做过%s检查吗？如果有，请回复1；如果没有，请回复2；如果不确定，请回复3'%(resp))] = 0
                return dict_data
            
        else:
            dict_data = {}
            dict_data[str('您好，请您按照实际情况回复1、2、3哦！')] = 0
            return dict_data

if __name__ == "__main__":
    app.run()