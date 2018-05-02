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
INumber = 50                       # 染色体个数 / 个体个数
BatchSize = 50                     # 寻找可用个体时用的批量上限
NumClasses = 1000                  # 标签种类
MaxEpoch = 1000                    # 迭代上限
Reserve = 0.1                      # 保留率 = 父子保留的精英量 / BestNumber
BestNmber = int(INumber*Reserve)   # 优秀样本数量
IndividualShape = (INumber,299,299,3)
Directions = 299*299*3
ImageShape = (299,299,3)
Sigma = 1
TopK = 5
Domin = 0.0
StartStdDeviation = 0.1
Convergence = 0.01
StartNumber = 2
Starteps = 0.5
EPSILON = 5e-2
MIN_EPS_DECAY = 5e-5
MAX_EPS_DECAY = 5e-3
UnVaildExist = 0            # 用来表示是否因为探索广度过大导致无效数据过多
def main():
    global OutDir
    global MaxEpoch
    global BatchSize
    global UnVaildExist
    global epsdecay
    QueryTimes = 0


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
        # L2Distance = tf.sqrt(tf.reduce_sum(tf.square(NewImage - SourceImg),axis=(1,2,3)))
        IndividualFitness = - (Sigma*tf.nn.softmax_cross_entropy_with_logits(logits=logit,labels=Labels))
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

        for p in range(5,100):
            # if p == 0:
            #     p = 5

            SSD = StartStdDeviation
            DM =Domin

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

            def StartPoint(sess, SImg, TImg, TargetClass,Domin):
                StartUpper = np.clip(TImg + Domin, 0.0, 1.0)
                StartDowner = np.clip(TImg - Domin, 0.0, 1.0)
                SImg = np.clip(SImg, StartDowner, StartUpper)
                return SImg

            StartImg = StartPoint(sess, SImg, TImg, TargetClass,DM)

            PBF = -1000000.0
            initI = np.zeros(IndividualShape, dtype = float)
            initCp = np.zeros((INumber,1000),dtype = float)
            ENP = np.zeros(ImageShape,dtype=float)
            DNP = np.zeros(ImageShape,dtype=float)+SSD
            LastGoodENP = ENP
            LastGoodDNP = DNP
            LastGoodEPS = Starteps
            realepsl = Starteps
            Lasteps = realepsl
            epsdecay = MAX_EPS_DECAY
            LogFile = open(os.path.join(OutDir, 'log%d.txt'%p), 'w+')
            QueryTimes = 0
            UnVaildExist = 0    # 用来表示是否因为探索广度过大导致无效数据过多
            CloseThreshold = -0.5
            DiffusionStd = 5e-3
            last_PBFS =[]
            Diffusiontimes =0
            Diffused = 0

            for i in range(MaxEpoch):
                Start = time.time()

                Upper = np.minimum(1.0 - StartImg, SImg + realepsl - StartImg)
                Downer = np.maximum(0.0 - StartImg, SImg - realepsl - StartImg)
                # 生成

                count = 0
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
                    QueryTimes += BatchSize
                    for j in range(BatchSize):
                        if TClass in CP[j].argsort()[-TopK:][::-1]:
                            initI[count] = temp[j]
                            initCp[count] = CP[j]
                            count += 1
                            if count == INumber:
                                break
                    if count!= INumber :
                        LogText = "count: %3d SSD: %.2f DM: %.3f p: %d"%(count,SSD,DM,p)
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

                    # 如果出现了样本无效化，回滚DNP,ENP
                    if i!=0 and count < StartNumber :
                        if epsdecay > MIN_EPS_DECAY:
                            epsdecay = max(epsdecay /2,MIN_EPS_DECAY)
                        realepsl = max(EPSILON,LastGoodEPS - epsdecay)
                        Upper = np.minimum(1.0 - StartImg, SImg + realepsl - StartImg)
                        Downer = np.maximum(0.0 - StartImg, SImg - realepsl - StartImg)
                        DNP = LastGoodDNP
                        ENP = LastGoodENP
                        LogText = "UnValidExist ANNEALING EPS DECAY"
                        LogFile.write(LogText + '\n')
                        print(LogText)

                    # 判断是否出现样本无效化
                    if cycletimes == 0:
                        if i != 0 and count < StartNumber:
                            UnVaildExist = 1
                        elif i != 0 and count >= StartNumber:
                            UnVaildExist = 0
                    cycletimes += 1

                initI = np.clip(initI,Downer,Upper)

                LastPBF,LastDNP,LastENP = PBF,DNP,ENP
                ENP,DNP,PBF,PB = sess.run([Expectation,StdDeviation,PbestFitness,Pbest],
                                          feed_dict={Individual: initI,logit:initCp,SourceImg:SImg,SourceClass:SClass,
                                                     TargetImg:TImg,TargetClass:TClass,Labels:LBS,StImg:StartImg})

                if PB.shape[0] > 1:
                    PB = PB[0]
                    PB = np.reshape(PB,(1,299,299,3))
                    print("PBConvergence")
                # render_frame(sess, GB, p*1000+i,SClass,TClass,StartImg)

                End = time.time()

                LogText = "Step %05d: PBF: %.4f realepsl: %.4f epsdecay: %.4f UseingTime: %.4f QueryTimes: %d" % (
                i, PBF,realepsl,epsdecay,End - Start,QueryTimes)
                LogFile.write(LogText + '\n')
                print(LogText)

                last_PBFS.append(PBF)
                last_PBFS = last_PBFS[-5:]

                if Diffused == 0 and (PBF < CloseThreshold) and Lasteps == realepsl and (PBF - LastPBF < Convergence or PBF < LastPBF):
                    DiffusionStd = DiffusionStd / 2
                    # DiffusionStd = epsdecay
                    DNP += DiffusionStd
                    print("Diffusion")
                    last_PBFS = []
                    Diffused = 1
                else :
                    Diffused = 0

                Lasteps = realepsl
                if DiffusionStd < MIN_EPS_DECAY:
                    if epsdecay > MIN_EPS_DECAY:
                        epsdecay = max(epsdecay / 2, MIN_EPS_DECAY)
                    realepsl = max(EPSILON, LastGoodEPS - epsdecay)
                    DNP = LastGoodDNP
                    ENP = LastGoodENP
                    last_PBFS = []
                    DiffusionStd = epsdecay
                    print("DiffusionStd < MIN_EPS_DECAY ANNEALING EPS DECAY")


                if (PBF > CloseThreshold):  # 靠近
                    LastGoodENP = ENP
                    DiffusionStd = epsdecay
                    DNP += DiffusionStd
                    LastGoodDNP = DNP
                    LastGoodEPS = realepsl
                    realepsl = max(EPSILON, realepsl - epsdecay)
                    epsdecay = MAX_EPS_DECAY
                    last_PBFS = []

                if PBF > CloseThreshold and realepsl <= EPSILON :
                    LogText = "Complete QueryTimes: %d"%(QueryTimes)
                    print(LogText)
                    LogFile.write(LogText + '\n')
                    render_frame(sess,PB, p, SClass, TClass, StartImg)
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

# if GB.shape[0] > 1:
#     GB = GB[0]
#     GB = np.reshape(GB, (1, 299, 299, 3))
#     # DNP += CDV
#     # ENP += (SImg - (StartImg + ENP)) * CEV
#     print("GBConvergence")

# if SSD >1 :
                    #     LogText = "Start Error"
                    #     LogFile.write(LogText + '\n')
                    #     print(LogText)
                    #     StartError = 1
                    #     break
                    # if count > 0:
                    #     LogText = "FindValidExample "
                    #     LogFile.write(LogText + '\n')
                    #     print(LogText)
                    #     FindValidExample = 1
                    #     break
# if i == 0 and count < StartNumber:
#     Times += 1
#     TimesUper = 1
#     if count > 0:
#         TimesUper = 5
#     else:
#         TimesUper = 1
#
#     if Times == TimesUper :
#         SSD += 0.01
#         if SSD - StartStdDeviation >= 0.05:
#             SSD = StartStdDeviation
#             DM -= 0.05
#             StartImg = StartPoint(sess, SImg, TImg, TargetClass,DM)
#             Upper = 1.0 - StartImg
#             Downer = 0.0 - StartImg
#
#         DNP = np.zeros(ImageShape,dtype=float) + SSD
#         Times = 0

# elif (Retry == 0 and Closed == 0 and Scaling == 0) and LastPBL2 - LastPBF < PBL2Distance - PBF:# 反而找到了不好的解
#     # Shaked = 1
#     DNP = LastDNP
#     ENP = LastENP
#     # LogText = "Shaked CEV: %.3f CDV: %.3f ConstantShaked: %2d" % (CEV, CDV,ConstantShaked)
#     LogText = "Shaked"
#     LogFile.write(LogText + '\n')
#     print(LogText)