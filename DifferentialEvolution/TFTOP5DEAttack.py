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
SourceClass = 0
TargetClass = 0
Sigma = 1
TopK = 5
Domin = 0.75
StartStdDeviation = 0.1
EVUper = 0.5
EVDown = 0.0001
CloseEVectorWeight = 0.3
CloseDVectorWeight = 0.01
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
        #

        def get_image(sess):
            global InputDir
            image_paths = sorted([os.path.join(InputDir, i) for i in os.listdir(InputDir)])

            index = np.random.randint(len(image_paths))
            path = image_paths[index]
            x = load_image(path)
            tempx = np.reshape(x, (1, 299, 299, 3))
            y = sess.run(GetP, {GetImage: tempx})
            y = y[0]
            return x, y

        SourceImg, SourceClass = get_image(sess)
        TargetImg, TargetClass = get_image(sess)
        while TargetClass == SourceClass:
            TargetImg, TargetClass = get_image(sess)

        StartUpper = np.clip(TargetImg + Domin, 0.0, 1.0)
        StartDowner = np.clip(TargetImg - Domin, 0.0, 1.0)

        one_hot_vec = one_hot(TargetClass, NumClasses)
        # ########################################计算startingpoint
        TestImge = tf.placeholder(shape=ImageShape,dtype=tf.float32) #（299，299，3）
        TestImgeEX = tf.reshape(TestImge, shape=(1, 299, 299, 3))  # （1，299，299，3）
        TestC, TestP = model(sess, TestImgeEX)  # TMD 为啥在我的程序里这个model不好使了

        def StartPoint(sess, SourceImg, TargetImg, TargetClass):
            SourceImg = np.clip(SourceImg, StartDowner, StartUpper)
            return SourceImg

        StartImg = StartPoint(sess, SourceImg, TargetImg, TargetClass)

        Upper = 1.0 - StartImg
        Downer = 0.0 - StartImg

        # ########################################startingpoint

        ## 预测
        GenI = tf.placeholder(shape=(BatchSize, 299, 299, 3), dtype=tf.float32)  # （299，299，3）
        GenC, GenP = model(sess, GenI)  # TMD 为啥在我的程序里这个model不好使了
        ##



        # 完成输入
        InputImg = tf.constant(StartImg,dtype=tf.float32) #（299，299，3）
        TempImg = tf.reshape(InputImg,shape=(1,299,299,3))   #（1，299，299，3）
        Labels = np.repeat(np.expand_dims(one_hot_vec, axis=0),repeats=INumber, axis=0) # （INumber，1000）

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

        GBF = -1000000.0
        PBF = GBF
        LastPBF = PBF

        GB = np.zeros([299,299,3],dtype=float)


        # ########################################计算输出结果

        def render_frame(sess, image, save_index):
            image = np.reshape(image,(299,299,3))+StartImg
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


        # init = tf.initialize_all_variables()!!!!!!这个会初始化全部的变量，包括其他函数里边的
        # sess.run(init)


        ##debug
        # initI = norm.rvs(loc=0, scale=0.1, size=IndividualShape)
        # C,P,I,N,E,D = sess.run([Confidence,Prediction,Individual,NewImage,Expectation,StdDeviation],feed_dict={Individual:initI})
        ##debug

        initI = np.zeros(IndividualShape, dtype = float)
        initCp = np.zeros((INumber,1000),dtype = float)
        ENP = np.zeros(ImageShape,dtype=float)
        DNP = ENP+StartStdDeviation
        LastENP = ENP
        LastDNP = DNP
        # GENP = np.zeros(ImageShape, dtype=float)
        # GDNP = ENP + 0.001
        LogFile = open(os.path.join(OutDir, 'log.txt'), 'w+')

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
                    if TargetClass in CP[j].argsort()[-TopK:][::-1]:
                        initI[count] = temp[j]
                        initCp[count] = CP[j]
                        count += 1
                        if count == INumber:
                            break
                if count!= INumber :
                    print("count: ", count," StartStdDeviation: ",StartStdDeviation)

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
                    if Times == 1 :
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

            initI = np.clip(initI,Downer,Upper)

            LastPBF = PBF
            LastDNP = DNP
            LastENP = ENP
            ENP,DNP,PBF,PB = sess.run([Expectation,StdDeviation,PbestFitness,Pbest],feed_dict={Individual: initI,logit:initCp})
            if PBF > GBF:
                GB = PB
                GBF = PBF



            if GB.shape[0] > 1:
                GB = GB[0]
                DNP += CloseDVectorWeight
                ENP += (SourceImg - (StartImg + ENP)) * CloseEVectorWeight
                print("GBConvergence")

            if PB.shape[0] > 1:
                PB = PB[0]
                PB = np.reshape(PB,(1,299,299,3))
            render_frame(sess, GB, i)

            End = time.time()
            GBL2Distance = np.sqrt(np.sum(np.square(StartImg + GB - SourceImg), axis=(1, 2, 3)))
            PBL2Distance = np.sqrt(np.sum(np.square(StartImg + PB - SourceImg), axis=(1, 2, 3)))

            LogText = "Step %05d: GBF: %.4f PBF: %.4f UseingTime: %.4f PBL2Distance: %.4f GBL2Distance: %.4f" % (
            i, GBF, PBF, End - Start, PBL2Distance,GBL2Distance)
            print(LogText)
            if UnVaildExist == 1 :#出现无效数据
                # CloseDVectorWeight /= 2
                CloseEVectorWeight -= 0.01
                DNP = LastDNP + CloseDVectorWeight
                ENP = LastENP + (SourceImg - (StartImg + ENP)) * CloseEVectorWeight
                print("UnValidExist CEV: ",CloseEVectorWeight)
            elif i>10 and LastPBF > PBF: # 发生抖动陷入局部最优(不应该以是否发生抖动来判断参数，而是应该以是否发现出现无效数据来判断，或者两者共同判断)
                # CloseDVectorWeight *= 2
                CloseEVectorWeight += 0.01
                # DNP += CloseDVectorWeight
                # ENP += (SourceImg - (StartImg + ENP)) * CloseEVectorWeight
                print("Shaked CEV: ",CloseEVectorWeight)

            if PBF - LastPBF < Convergence and LastPBF < PBF and UnVaildExist!=1:#不能重复靠近
                if GBL2Distance < 10:
                    print("Complete")
                    LogFile.write(LogText + '\n')
                    break
                DNP += CloseDVectorWeight
                ENP += (SourceImg-(StartImg+ENP))*CloseEVectorWeight
                print("Close up CEV: ",CloseEVectorWeight)


            # if GBF > -13:
            #     np.save(os.path.join(OutDir, '13E.npy', ENP))
            #     np.save(os.path.join(OutDir, '13D.npy', DNP))
            LogFile.write(LogText+'\n')


            if CloseEVectorWeight < EVDown:
                break


# def get_image(index):
#     global InputDir
#     data_path = os.path.join(InputDir, 'val')
#     image_paths = sorted([os.path.join(data_path, i) for i in os.listdir(data_path)])
#     # 修改
#     # assert len(image_paths) == 50000
#     labels_path = os.path.join(InputDir, 'val.txt')
#     with open(labels_path) as labels_file:
#         labels = [i.split(' ') for i in labels_file.read().strip().split('\n')]
#         labels = {os.path.basename(i[0]): int(i[1]) for i in labels}
#
#     def get(index):
#         path = image_paths[index]
#         x = load_image(path)
#         y = labels[os.path.basename(path)]
#         return x, y
#
#     return get(index)


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
