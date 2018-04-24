# 本程序为差分进化算法实现攻击的demo
# 问题：
# 1. 原始图片都不能正常识别
# 2. 生成10^7个随机数，耗时过长
# in this version, we meet some trouble
import PIL
from PIL import Image
from inception_v3_imagenet import model, SIZE
import matplotlib.pyplot as plt
from utils import *
from scipy.stats import norm
from imagenet_labels import label_to_name

import numpy as np
import tensorflow as tf
import os
import sys
import shutil
import time
import scipy


os.environ["CUDA_VISIBLE_DEVICES"] = "2"

InputDir = "adv_samples/"
OutDir = "adv_example/"
SourceIndex = 0
TargetIndex = 1
INumber = 100               # 染色体个数 / 个体个数
BatchSize = 100             # 寻找可用个体时用的批量上限
NumClasses = 1000           # 标签种类
MaxEpoch = 10000            # 迭代上限
Reserve = 0.25              # 保留率 = 父子保留的精英量 / BestNumber
BestNmber = int(INumber*Reserve) # 优秀样本数量
IndividualShape = (INumber,299,299,3)
Directions = 299*299*3
ImageShape = (299,299,3)
Sigma = 1
TopK = 5
Domin = 0.25
StartStdDeviation = 0.01
CloseEVectorWeight = 0.3
CloseDVectorWeight = 0.02
Convergence = 0.01
StartNumber = 2
Closed = 0                  # 用来标记是否进行靠近操作
UnVaildExist = 0            # 用来表示是否因为探索广度过大导致无效数据过多
def main():
    global OutDir
    global MaxEpoch
    global SourceClass
    global TargetClass
    global StartStdDeviation
    global BatchSize
    global CloseDVectorWeight
    global CloseEVectorWeight
    global UnVaildExist


    if os.path.exists(OutDir):
        shutil.rmtree(OutDir)
    os.makedirs(OutDir)


    with tf.Session() as sess:

        # get image label
        GetImage = tf.placeholder(shape=(1, 299, 299, 3), dtype=tf.float32)  # （299，299，3）
        GetC, GetP = model(sess, GetImage)  # TMD 为啥在我的程序里这个model不好使了
        # get image label

        # ########################################计算render_frame
        TestImge = tf.placeholder(shape=ImageShape,dtype=tf.float32) #（299，299，3）
        TestImgeEX = tf.reshape(TestImge, shape=(1, 299, 299, 3))  # （1，299，299，3）
        TestC, TestP = model(sess, TestImgeEX)

        # ########################################render_frame

        ## 预测get bath image label
        GenI = tf.placeholder(shape=(BatchSize, 299, 299, 3), dtype=tf.float32)  # （299，299，3）
        GenC, GenP = model(sess, GenI)
        ## 预测get bath image label

        SourceImg = tf.placeholder(dtype=tf.float32,shape=(299,299,3))
        SourceClass = tf.placeholder(dtype=tf.int32)
        TargetImg = tf.placeholder(dtype=tf.float32,shape=(299,299,3))
        TargetClass = tf.placeholder(dtype=tf.int32)

        # one_hot_vec = tf.placeholder(dtype=tf.float32,shape=(1000))
        StImg = tf.placeholder(dtype=tf.float32,shape=(299,299,3))


        # 完成输入
        InputImg = StImg #（299，299，3）
        TempImg = tf.reshape(InputImg,shape=(1,299,299,3))   #（1，299，299，3）
        # Labels = tf.reshape(tf.tile(one_hot_vec,[INumber]), (INumber,1000)) # （INumber，1000）
        Labels = tf.placeholder(dtype=tf.int32,shape=(INumber,1000))

        # 开始进化算法
        Individual = tf.placeholder(shape=IndividualShape,dtype=tf.float32) # （INumber，299，299，3）
        Expectation = tf.Variable(np.random.random([299,299,3]),dtype=tf.float32) # （299，299，3）
        Deviation = tf.Variable(np.random.random([299,299,3]),dtype=tf.float32) # （299，299，3）

        NewImage = Individual + TempImg

        # 计算置信度
        # logit,pred = model(sess,NewImage) # TMD 为啥在我的程序里这个model不好使了
        logit = tf.placeholder(shape=(INumber,1000),dtype=tf.float32)
        # TOPK局部信息


        # 计算适应度
        # IndividualFitness = -tf.reduce_sum(Labels * tf.log(Confidence), 1) #（INumber）
        # （INumber，1）还是（INumber） ？ 是（INumber）
        # reduction_indices 表示求和方向，并降维
        L2Distance = tf.sqrt(tf.reduce_sum(tf.square(NewImage - SourceImg),axis=(1,2,3)))
        IndividualFitness = - (Sigma*tf.nn.softmax_cross_entropy_with_logits(logits=logit,labels=Labels)+ L2Distance)
        # IndividualFitness = - (Sigma*(-tf.reduce_sum(Labels * tf.log(Confidence), 1))) # （INumber）


        # 选取优秀的的前BestNmber的个体
        TopKFit,TopKFitIndx =  tf.nn.top_k(IndividualFitness,BestNmber)
        TopKIndividual = tf.gather(Individual,TopKFitIndx) # (BestNmber,299,299,3) 此处是否可以完成

        # 更新期望与方差
        Expectation = tf.constant(np.zeros(ImageShape),dtype=tf.float32)
        for i in range(BestNmber):
            Expectation += (0.5**(i+1)*TopKIndividual[i])
        # Expectation = tf.reduce_mean(TopKIndividual,reduction_indices=0)
        Deviation = tf.constant(np.zeros(ImageShape),dtype=tf.float32)
        for i in range(BestNmber):
            Deviation += 0.5**(i+1)*tf.square(TopKIndividual[i] - Expectation)
        # Deviation /= BestNmber
        StdDeviation = tf.sqrt(Deviation)

        # 获取种群最佳（活着的，不算历史的）
        PbestFitness = tf.reduce_max(IndividualFitness)
        Pbestinds = tf.where(tf.equal(PbestFitness,IndividualFitness))
        Pbestinds = Pbestinds[:,0]
        Pbest = tf.gather(Individual,Pbestinds)

        # ########################################计算输出结果

        def render_frame(sess, image, save_index,SourceClass,TargetClass,StartImg):
            image = np.reshape(image,(299,299,3))+StartImg
            scipy.misc.imsave(os.path.join(OutDir, '%s.jpg' % save_index), image)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
            # image
            ax1.imshow(image)
            fig.sca(ax1)
            plt.xticks([])
            plt.yticks([])
            # classifications
            probs = softmax(sess.run(TestC, {TestImge: image})[0])
            topk = probs.argsort()[-5:][::-1]
            topprobs = probs[topk]
            barlist = ax2.bar(range(5), topprobs)
            for i, v in enumerate(topk):
                if v == SourceClass:
                    barlist[i].set_color('g')
                if v == TargetClass:
                    barlist[i].set_color('r')
            plt.sca(ax2)
            plt.ylim([0, 1.1])
            plt.xticks(range(5), [label_to_name(i)[:15] for i in topk], rotation='vertical')
            fig.subplots_adjust(bottom=0.2)

            path = os.path.join(OutDir, 'frame%06d.png' % save_index)
            if os.path.exists(path):
                os.remove(path)
            plt.savefig(path)
            plt.close()
        # #################################################3计算输出结果

        def get_image(sess,indextemp=-1):
            global InputDir
            image_paths = sorted([os.path.join(InputDir, i) for i in os.listdir(InputDir)])

            if indextemp != -1:
                index = indextemp
            else:
                index = np.random.randint(len(image_paths))

            path = image_paths[index]
            x = load_image(path)
            tempx = np.reshape(x, (1, 299, 299, 3))
            y = sess.run(GetP, {GetImage: tempx})
            y = y[0]
            return x, y

        for p in range(100):
            if p == 0:
                p = 5
            StartStdDeviation = 0.01
            CloseEVectorWeight = 0.3
            CloseDVectorWeight = 0.02
            Closed = 0  # 用来标记是否进行靠近操作
            UnVaildExist = 0  # 用来表示是否因为探索广度过大导致无效数据过多

            index1 = p//10
            index2 = p%10

            SImg, SClass = get_image(sess,index1)
            TImg, TClass = get_image(sess,index2)
            OHV= one_hot(TClass,NumClasses)
            LBS = np.repeat(np.expand_dims(OHV, axis=0),repeats=INumber, axis=0)
            if TClass == SClass:
                LogText = "SClass == TClass"
                LogFile = open(os.path.join(OutDir, 'log%d.txt' % p), 'w+')
                LogFile.write(LogText + '\n')
                print(LogText)
                continue

            StartUpper = np.clip(TImg + Domin, 0.0, 1.0)
            StartDowner = np.clip(TImg - Domin, 0.0, 1.0)

            def StartPoint(sess, SImg, TImg, TargetClass):
                SImg = np.clip(SImg, StartDowner, StartUpper)
                return SImg

            StartImg = StartPoint(sess, SImg, TImg, TargetClass)
            Upper = 1.0 - StartImg
            Downer = 0.0 - StartImg

            GBF = -1000000.0
            PBF = GBF
            LastPBF = PBF
            GB = np.zeros([299, 299, 3], dtype=float)

            initI = np.zeros(IndividualShape, dtype = float)
            initCp = np.zeros((INumber,1000),dtype = float)
            ENP = np.zeros(ImageShape,dtype=float)
            DNP = ENP+StartStdDeviation
            LastENP = ENP
            LastDNP = DNP
            LogFile = open(os.path.join(OutDir, 'log%d.txt'%p), 'w+')
            PBL2Distance = 100000
            LastPBL2 = PBL2Distance
            StartError = 0

            for i in range(MaxEpoch):
                Start = time.time()

                # 生成
                count = 0
                Times = 0
                cycletimes = 0
                while count != INumber:

                # 制造
                    DNPT = np.reshape(DNP,(1,299,299,3))
                    ENPT = np.reshape(ENP,(1,299,299,3))

                    temp = np.random.randn(BatchSize,299,299,3)
                    temp = temp*DNPT + ENPT
                    temp = np.clip(temp, Downer, Upper)

                    testimage = temp + np.reshape(StartImg,(1,299,299,3))
                    CP, PP = sess.run([GenC,GenP],{GenI:testimage})
                    CP = np.reshape(CP, (BatchSize, 1000))
                # 筛选
                    for j in range(BatchSize):
                        if TClass in CP[j].argsort()[-TopK:][::-1]:
                            initI[count] = temp[j]
                            initCp[count] = CP[j]
                            count += 1
                            if count == INumber:
                                break
                    if count!= INumber :
                        LogText = "count: %3d StartStdDeviation: %.2f"%(count,StartStdDeviation)
                        LogFile.write(LogText + '\n')
                        print(LogText)

                    if count > StartNumber-1 and count < INumber :
                        tempI = initI[0:count]
                        ENP = np.zeros(ImageShape, dtype=float)
                        DNP = np.zeros(ImageShape, dtype=float)
                        for j in range(count):
                            ENP += tempI[j]
                        ENP /= count
                        for j in range(count):
                            DNP += np.square(tempI[j] - ENP)
                        DNP /= count
                        DNP = np.sqrt(DNP)

                    if i == 0 and count < StartNumber:
                        Times += 1
                        if Times == 5 :
                            StartStdDeviation += 0.01
                            DNP = ENP + StartStdDeviation
                            Times = 0

                    if i>10 and count < INumber/25 :
                        DNP = LastDNP
                        ENP = LastENP

                    if cycletimes == 0:
                        if i > 10 and count < INumber/25:
                            UnVaildExist = 1
                        elif i > 10 and count >= INumber/25:
                            UnVaildExist = 0
                    cycletimes += 1

                    if StartStdDeviation >1 :
                        LogText = "Start Error"
                        LogFile.write(LogText + '\n')
                        print(LogText)
                        StartError = 1
                        break
                if StartError == 1:
                    break

                initI = np.clip(initI,Downer,Upper)

                LastPBF,LastDNP,LastENP = PBF,DNP,ENP
                ENP,DNP,PBF,PB = sess.run([Expectation,StdDeviation,PbestFitness,Pbest],
                                          feed_dict={Individual: initI,logit:initCp,SourceImg:SImg,SourceClass:SClass,
                                                     TargetImg:TImg,TargetClass:TClass,Labels:LBS,StImg:StartImg})
                if PBF > GBF:
                    GB = PB
                    GBF = PBF

                if GB.shape[0] > 1:
                    GB = GB[0]
                    B = np.reshape(B, (1, 299, 299, 3))
                    # DNP += CloseDVectorWeight
                    # ENP += (SImg - (StartImg + ENP)) * CloseEVectorWeight
                    print("GBConvergence")

                if PB.shape[0] > 1:
                    PB = PB[0]
                    PB = np.reshape(PB,(1,299,299,3))
                    print("PBConvergence")
                # render_frame(sess, GB, p*1000+i,SClass,TClass,StartImg)


                End = time.time()
                GBL2Distance = np.sqrt(np.sum(np.square(StartImg + GB - SImg), axis=(1, 2, 3)))
                LastPBL2 = PBL2Distance
                PBL2Distance = np.sqrt(np.sum(np.square(StartImg + PB - SImg), axis=(1, 2, 3)))

                LogText = "Step %05d: GBF: %.4f PBF: %.4f UseingTime: %.4f PBL2Distance: %.4f GBL2Distance: %.4f" % (
                i, GBF, PBF, End - Start, PBL2Distance,GBL2Distance)
                LogFile.write(LogText + '\n')
                print(LogText)
                if UnVaildExist == 1 :#出现无效数据
                    # CloseDVectorWeight /= 2
                    CloseEVectorWeight -= 0.01
                    # CloseDVectorWeight += 0.001
                    DNP = LastDNP + CloseDVectorWeight
                    ENP = LastENP + (SImg - (StartImg + ENP)) * CloseEVectorWeight
                    LogText = "UnValidExist CEV: %.3f CDV: %.3f"%(CloseEVectorWeight,CloseDVectorWeight)
                    LogFile.write(LogText + '\n')
                    print(LogText)
                # elif i>10 and LastPBF > PBF: # 发生抖动陷入局部最优(不应该以是否发生抖动来判断参数，而是应该以是否发现出现无效数据来判断，或者两者共同判断)
                elif Closed == 1:
                    if LastPBL2 < PBL2Distance:
                        # CloseDVectorWeight -= 0.001
                        LogText = "L2Shaked CEV: %.3f CDV: %.3f" % (CloseEVectorWeight, CloseDVectorWeight)
                        LogFile.write(LogText + '\n')
                        print(LogText)
                    # CloseDVectorWeight *= 2
                    CloseEVectorWeight += 0.01
                    # DNP += CloseDVectorWeight
                    # ENP += (SourceImg - (StartImg + ENP)) * CloseEVectorWeight
                    LogText = "CEVUP CEV: %.3f CDV: %.3f"%(CloseEVectorWeight,CloseDVectorWeight)
                    LogFile.write(LogText + '\n')
                    print(LogText)


                Closed = 0
                if PBF - LastPBF < Convergence and LastPBF < PBF and UnVaildExist!=1:#不能重复靠近
                    if GBL2Distance < 25:
                        print("Complete")
                        LogFile.write(LogText + '\n')
                        render_frame(sess, GB, p, SClass, TClass, StartImg)
                        break
                    DNP += CloseDVectorWeight
                    ENP += (SImg-(StartImg+ENP))*CloseEVectorWeight
                    Closed = 1
                    LogText = "Close up CEV: %.3f CDV: %.3f" % (CloseEVectorWeight, CloseDVectorWeight)
                    LogFile.write(LogText + '\n')
                    print(LogText)

                if CloseEVectorWeight <= 0 or CloseDVectorWeight <= 0:
                    LogText = "Closed Error"
                    LogFile.write(LogText + '\n')
                    print(LogText)
                    break


def load_image(path):
    image = PIL.Image.open(path)
    if image.height > image.width:
        height_off = int((image.height - image.width) / 2)
        image = image.crop((0, height_off, image.width, height_off + image.width))
    elif image.width > image.height:
        width_off = int((image.width - image.height) / 2)
        image = image.crop((width_off, 0, width_off + image.height, image.height))
    image = image.resize((299, 299))
    img = np.asarray(image).astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.repeat(img[:, :, np.newaxis], repeats=3, axis=2)
    if img.shape[2] == 4:
        # alpha channel
        img = img[:, :, :3]
    return img


if __name__ == '__main__':
    main()
