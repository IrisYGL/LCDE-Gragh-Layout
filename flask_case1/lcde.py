from flask import Flask,render_template,request
import numpy as np
import json
import math
import time
from numpy import random
import scipy
from scipy import linalg
#import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/search', methods=['GET', 'POST'])
def search():
    #-----------------传入参数
    subnum_init = request.form.getlist("subnum")
    id_init = request.form.get("node_id")
    x_init = request.form.get("node_x")
    y_init = request.form.get("node_y")
    edges_init = request.form.get("total_edges")
    source_init = request.form.get("source")
    target_init = request.form.get("target")
    initialx_init = request.form.get("initialx")
    initialy_init = request.form.get("initialy")
    subnum = int(subnum_init[0]) #子图个数
    id = json.loads(id_init) #二维数组，id[n][i]表示第n个子图中第i个点的id
    x = json.loads(x_init) #二维数组，x[n][i]表示第n个子图中第i个点的x坐标
    y = json.loads(y_init) #二维数组，x[n][i]表示第n个子图中第i个点的y坐标
    edges = json.loads(edges_init) #初始大图中的所有的边
    source = json.loads(source_init)
    target = json.loads(target_init)
    initialx1 = json.loads(initialx_init)
    initialx = np.mat(json.loads(initialx_init))
    initialy = np.mat(json.loads(initialy_init))

    
    #-----------------矩阵构建
    Res0 = np.hstack((initialx.T,initialy.T))
    
    count = 0
    for pair in edges:
        count += 1;
    insub = [0]*count
    subadd = []

    r = 0 #矩阵的秩，所有子图不同的id的个数
    r = len(initialx1)
    times = np.mat(np.zeros((r,r)))
    dis = np.mat(np.zeros((r,r)))
    d = np.mat(np.zeros((r,r)))
    Lw = np.mat(np.zeros((r,r)))
    Lsw = np.mat(np.zeros((r,r)))

    k = -1
    for sub in id:
        k += 1
        for ii in range(len(id[k])):
            i = id[k][ii]
            for jj in range(len(id[k])):
                j = id[k][jj]
                if i!=j and len(x[k]) != 0:
                    dis[i,j] += math.sqrt(pow((x[k][ii] - x[k][jj]),2) + pow((y[k][ii] - y[k][jj]),2))
                    times[i,j] += 1

    count2 = -1
    for pair in edges:
        count2 += 1
        i = pair[0]
        j = pair[1]
        for sub in id:
            if i in sub and j in sub:
                insub[count2] += 1

    for ii in range(len(edges)):
        if insub[ii] == 0:
            i = edges[ii][0]
            j = edges[ii][1]
            dis[i,j] += math.sqrt(pow((Res0[i,0] - Res0[j,0]),2) + pow((Res0[i,1] - Res0[j,1]),2))
            dis[j,i] += math.sqrt(pow((Res0[i,0] - Res0[j,0]),2) + pow((Res0[i,1] - Res0[j,1]),2))
            times[i,j] += 1
            times[j,i] += 1

    for i in range(r):     #Lw和Lsw对角线
        for j in range(r):
            if times[i,j] != 0:
                d[i,j] = dis[i,j] / times[i,j]

    k = -1
    for sub in id:
        k += 1
        for ii in range(len(id[k])):
            i = id[k][ii]
            for jj in range(len(id[k])):
                j = id[k][jj]
                if i!=j and len(x[k]) != 0:
                    Lsw[i,j] = - 1 / pow(d[i,j],2)

    for ii in range(len(edges)):
        i = edges[ii][0]
        j = edges[ii][1]
        if d[i,j]!=0:
            Lw[i,j] = - 1 / pow(d[i,j],2)
            Lw[j,i] = - 1 / pow(d[j,i],2)

    for i in range(r):     #Lw和Lsw对角线
        for j in range(r):
            if i==j:
                Lsw[i,j] = -np.sum(Lsw[i,:])
                Lw[i,j] = -np.sum(Lw[i,:])

    #-----------------迭代求解
#    Res0 = np.random.randint(1,1000,[r,2]) #随机生成迭代初始矩阵
    Res0 = np.hstack((initialx.T,initialy.T))
    Res0[0,0] = 0
    Res0[0,1] = 0
    Check = np.mat(np.zeros((r,2)))
    Res = np.random.randint(1,1000,[r,2])
    ResTemp = np.random.randint(1,1000,[r-1,2])
    Top = np.mat(np.zeros((1,2)))
    alpha = 1 #alpha参数值
    left = Lw + alpha * Lsw
    left1 = np.delete(left,0,axis=1)
    LI = np.linalg.pinv(left1)
    iteration = 0
    Temp = Res0
    CheckInt = 100
    time_start=time.time()
    while CheckInt > 0.01:
        iteration += 1
        Lwd = np.mat(np.zeros((r,r)))
        Lswd = np.mat(np.zeros((r,r)))
        k = -1
        for sub in id:
            k += 1
            for ii in range(len(id[k])):
                i = id[k][ii]
                for jj in range(len(id[k])):
                    j = id[k][jj]
                    if i!=j and len(x[k]) != 0:
                        Lswd[i,j] = Lsw[i,j] * d[i,j] / math.sqrt(pow((Temp[i,0] - Temp[j,0]),2) + pow((Temp[i,1] - Temp[j,1]),2))
        for ii in range(len(edges)):
            i = edges[ii][0]
            j = edges[ii][1]
            if d[i,j] != 0:
                Lwd[i,j] = Lw[i,j] * d[i,j] / math.sqrt(pow((Temp[i,0] - Temp[j,0]),2) + pow((Temp[i,1] - Temp[j,1]),2))
                Lwd[j,i] = Lw[j,i] * d[j,i] / math.sqrt(pow((Temp[i,0] - Temp[j,0]),2) + pow((Temp[i,1] - Temp[j,1]),2))
        for i in range(r):     #Lw和Lsw对角线
            for j in range(r):
                if i==j:
                    Lswd[i,j] = -np.sum(Lswd[i,:])
                    Lwd[i,j] = -np.sum(Lwd[i,:])
        ResTemp = LI * (Lwd + alpha * Lswd) * Temp
        Res = np.r_[Top,ResTemp]
        Check = Res - Temp
        CheckInt = abs(np.sum(Check))
        print(CheckInt)
        Temp = Res #迭代右边的值赋给左边

    #-----------------衡量相似度
    k = -1
    for sub in id:
        listx_init = []
        listy_init = []
        listx = []
        listy= []
        k += 1
        if len(x[k]) != 0:
            xx1 = []
            yy1 = []
            xx2 = []
            yy2 = []
            for ii in range(len(id[k])):
                i = id[k][ii]
                listx_init.append(x[k][ii])
                listy_init.append(y[k][ii])
                xx1.append(x[k][ii])
                yy1.append(y[k][ii])
                if i==0:
                    listx.append(Res[i,0])
                    listy.append(Res[i,1])
                    xx2.append(x[k][ii])
                    yy2.append(y[k][ii])
                else:
                    listx.append(Res[i,0])
                    listy.append(Res[i,1])
                    xx2.append(x[k][ii])
                    yy2.append(y[k][ii])
            listx_init1 = np.mat(listx_init)
            listy_init1 = np.mat(listy_init)
            listx1 = np.mat(listx)
            listy1 = np.mat(listy)
            X = np.hstack((listx_init1.T,listy_init1.T))
            Y = np.hstack((listx1.T,listy1.T))
            print(X)
            print(Y)
            print('x1=',xx1)
            print('y1=',yy1)
            print('x2=',xx2)
            print('y2=',yy2)
            a = pow((np.trace(scipy.linalg.sqrtm(X.T*Y*Y.T*X))),2)
            b = (np.trace(X.T*X))*(np.trace(Y.T*Y))
#            print(a,b)
            similarity = a/b
            print('similarity',similarity)

    #-----------------画图
    time_end=time.time()
#    print('time',time_end-time_start)
#    print('interation',iteration)
    xx = []
    yy = []
    for i in range(r):
        xx.append(Res[i,0])
        yy.append(Res[i,1])
    print('x=',xx)
    print('y=',yy)
    print('source=',source)
    print('target=',target)
    return str(Res)

if __name__ == '__main__':
    app.run()
