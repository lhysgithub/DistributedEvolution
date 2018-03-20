import tensorflow as tf
import numpy as np
import math

class GAOptimizer(object):
    def __init__(self,sess,imagesize,loss,BatchSize,max_iterations=100,chromosomesNumber=5,k1=1.0,k2=1.0,k3=0.5,k4=0.5):
        lenth = imagesize*imagesize
        self.MAX_ITERATIONS = max_iterations
        self.ChromosomesNumber = chromosomesNumber
        self.ChromosomesLenth = lenth
        self.k1=k1
        self.k2=k2
        self.k3=k3
        self.k4=k4
        self.PI=3.1415926

        self.ChromosomesFitness = np.zeros(self.ChromosomesNumber)
        self.ChromosomesPC = np.random.random(BatchSize*self.ChromosomesNumber).reshape(self.ChromosomesNumber,BatchSize)
        self.ChromosomesPM = np.random.random(BatchSize*self.ChromosomesNumber).reshape(self.ChromosomesNumber,BatchSize)
        self.ChromosomesMaxFitness = np.zeros(loss.shape)
        self.ChromosomesAvgFitness = np.zeros(loss.shape)
        self.ChromosomesSumFitness = np.zeros(loss.shape)
        self.WaitCross = []*BatchSize
        self.sort= np.zeros(BatchSize*chromosomesNumber).reshape(BatchSize,chromosomesNumber)
        # self.ChromosomesNewImg = np.array([np.zeros(lenth)]*self.ChromosomesNumber, dtype = np.float32)
        # self.ChromosomesOutPut = np.zeros(self.ChromosomesNumber)
        # self.ChromosomesL2dist = np.zeros(self.ChromosomesNumber)
    def minimize(self,loss,modifier,Imgs,Labs,Consts,boxmul,boxplus,model,TARGETED,CONFIDENCE):
        self.Chromosomes = [np.random.random(modifier.shape)]* self.ChromosomesNumber
        self.ChromosomesFitness = [np.zeros(loss.shape)] * self.ChromosomesNumber
        # 更新适应度并选最优
        for i in range(self.ChromosomesNumber):
            # 怎么获取单个染色体？

            self.ChromosomesOutPut = model.predict((self.Chromosomes[i] + Imgs) * boxmul + boxplus)
            self.ChromosomesL2dist = tf.reduce_sum(tf.square(((self.Chromosomes[i] + Imgs) * boxmul + boxplus) - ((Imgs) * boxmul + boxplus)),[1, 2, 3])
            real = tf.reduce_sum((Labs) * self.ChromosomesOutPut, 1)
            other = tf.reduce_max((1 - Labs) * self.ChromosomesOutPut - (Labs * 10000), 1)

            if TARGETED:  # if targetted, optimize for making the other class most likely
                loss1 = tf.maximum(0.0, other - real + CONFIDENCE)
            else:  # if untargeted, optimize for making this class least likely.
                loss1 = tf.maximum(0.0, real - other + CONFIDENCE)

            # sum up the losses
            self.loss2 = tf.reduce_sum(self.ChromosomesL2dist)
            self.loss1 = tf.reduce_sum(Consts * loss1)
            self.ChromosomesFitness[i] = self.loss1 + self.loss2
            self.ChromosomesSumFitness += self.ChromosomesFitness[i]
            self.sort[i]=i
            loss = tf.where(tf.less(self.ChromosomesFitness[i],loss),self.ChromosomesFitness[i],loss)
            self.ChromosomesMaxFitness = tf.where(tf.less(self.ChromosomesFitness[i], loss), self.ChromosomesFitness[i], self.ChromosomesMaxFitness)
            modifier = tf.where(tf.less(self.ChromosomesFitness[i], loss), self.Chromosomes[i], modifier)

        self.ChromosomesAvgFitness = self.ChromosomesSumFitness / self.ChromosomesNumber


        # 涉及batchsize，待改
        # 按照适应度排序染色体
        for i in range(self.ChromosomesNumber-1):
            for j in range(self.ChromosomesNumber-i-1):
                if(self.ChromosomesFitness[self.sort[j]]<self.ChromosomesFitness[self.sort[j+1]]):
                    temp = self.sort[j]
                    self.sort[j] = self.sort[j+1]
                    self.sort[j + 1] = temp

        # 轮盘赌选杂交池
        for i in range(self.ChromosomesNumber):
            if self.ChromosomesFitness[i] > self.ChromosomesAvgFitness :
                self.ChromosomesPC[i] = self.k1*math.sin(self.PI*(self.ChromosomesMaxFitness-self.ChromosomesFitness[i])/(2*(self.ChromosomesMaxFitness-self.ChromosomesAvgFitness)))
            else:
                self.ChromosomesPC[i] = self.k2
            if(1-np.random.random() < self.ChromosomesPC):
                self.WaitCross.append(self.ChromosomesPC[i])
        #杂交
        temps =[]
        while len(self.WaitCross)>=2 :
            x1 = np.random.randint(len(self.WaitCross))
            temp1 = self.WaitCross[x1]
            self.WaitCross[x1]=self.WaitCross[len(self.WaitCross)-1]
            self.WaitCross.pop()
            x2 = np.random.randint(len(self.WaitCross))
            temp2 = self.WaitCross[x2]
            self.WaitCross[x2] = self.WaitCross[len(self.WaitCross) - 1]
            temp3 = temp1[self.ChromosomesLenth/2:]
            temp1[self.ChromosomesLenth/2:] = temp2[self.ChromosomesLenth/2:]
            temp2[self.ChromosomesLenth/2:] = temp3
            temps.append(temp1)
            temps.append(temp2)

        # 突变
        for i in range(self.ChromosomesNumber):
            if self.ChromosomesFitness[i] > self.ChromosomesAvgFitness :
                self.ChromosomesPM[i] = self.k3*math.sin(self.PI*(self.ChromosomesMaxFitness-self.ChromosomesFitness[i])/(2*(self.ChromosomesMaxFitness-self.ChromosomesAvgFitness)))
            else:
                self.ChromosomesPM[i] = self.k4
            if(1-np.random.random() < self.ChromosomesPM):
                for j in range(self.ChromosomesLenth):
                    if(1-np.random.random() < self.ChromosomesPM):
                        self.Chromosomes[i][j]=1-np.random.random()

        # 补全染色体组
        count = 0
        while len(temps)<self.ChromosomesLenth:
            temps.append(self.Chromosomes[self.sort[count]])
            count += 1
        self.Chromosomes[:] = temps

        return modifier




