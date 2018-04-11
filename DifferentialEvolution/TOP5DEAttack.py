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
INumber = 200               # 染色体个数 / 个体个数
BatchSize = 400             # 寻找可用个体时用的批量上限
NumClasses = 1000           # 标签种类
MaxEpoch = 1000             # 迭代上限
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
VectorWeight = 0.0125
Convergence = 0.01
def main():
    global OutDir
    global MaxEpoch
    global SourceClass
    global TargetClass
    global StartStdDeviation

    SourceImg,SourceClass = get_image(SourceIndex)
    TargetImg, TargetClass = get_image(TargetIndex)


    StartUpper = np.clip(TargetImg + Domin,0.0,1.0)
    StartDowner = np.clip(TargetImg - Domin,0.0,1.0)

    if os.path.exists(OutDir):
        shutil.rmtree(OutDir)
    os.makedirs(OutDir)
    one_hot_vec = one_hot(TargetClass, NumClasses)

    with tf.Session() as sess:

        # ########################################计算startingpoint
        TestImge = tf.placeholder(shape=ImageShape,dtype=tf.float32) #（299，299，3）
        TestImgeEX = tf.reshape(TestImge, shape=(1, 299, 299, 3))  # （1，299，299，3）
        TestC, TestP = model(sess, TestImgeEX)  # TMD 为啥在我的程序里这个model不好使了

        def StartPoint(sess, SourceImg, TargetImg, TargetClass):
            SourceImg = np.clip(SourceImg, StartDowner, StartUpper)
            # for i in range(100):
            #     InputImg = (TargetImg - SourceImg) * 0.01 * i
            #     TC, TP = sess.run([TestC, TestP], {TestImge: InputImg})
            #     if TargetClass in TC[0].argsort()[-TopK:][::-1]:
            #         return InputImg
            #     else:
            #         continue
            return SourceImg

        StartImg = StartPoint(sess, SourceImg, TargetImg, TargetClass)

        Upper = 1.0 - StartImg
        Downer = 0.0 - StartImg

        # ########################################startingpoint

        ## 预测
        GenI = tf.placeholder(shape=(BatchSize,299,299,3),dtype=tf.float32) #（299，299，3）
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
        # Confidenceplaceholder = tf.placeholder(dtype=tf.float32)
        # Predictiondsplaceholder = tf.placeholder(dtype=tf.int32)
        # Confidence = Confidenceplaceholder # （INumber，K）
        # Prediction = Predictiondsplaceholder # （INumber）
        logit,pred = model(sess,NewImage) # TMD 为啥在我的程序里这个model不好使了
        # Confidenceplaceholder = tf.placeholder(dtype=tf.float32)
        # Predictiondsplaceholder = tf.placeholder(dtype=tf.int32)
        # Confidence = Confidenceplaceholder # （INumber，K）
        # Prediction = Predictiondsplaceholder # （INumber）

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
            # testmax = np.max(image)
            # testmin = np.min(image)
            image = np.reshape(image,(299,299,3))+StartImg
            # testmax = np.max(image)
            # testmin = np.min(image)
            # actually draw the figure
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

        initI = np.zeros(IndividualShape, dtype=float)
        ENP = np.zeros(ImageShape,dtype=float)
        DNP = ENP+StartStdDeviation
        # GENP = np.zeros(ImageShape, dtype=float)
        # GDNP = ENP + 0.001
        LogFile = open(os.path.join(OutDir, 'log.txt'), 'w+')

        for i in range(MaxEpoch):
            Start = time.time()

            # 生成
            count = 0
            Times = 0
            while count != INumber:
                # 制造

                temp = np.zeros((3, 299, 299, BatchSize), dtype=float)
                for j in range(3):
                    for k in range(299):
                        for l in range(299):
                            temp[j][k][l] = norm.rvs(loc=ENP[l][k][j], scale=DNP[l][k][j], size=BatchSize)
                temp = temp.transpose((3, 2, 1, 0))
                temp = np.clip(temp, Downer, Upper)
                testimage = temp + np.reshape(StartImg,(1,299,299,3))
                CP, PP = sess.run([GenC,GenP],{GenI:testimage})
                CP = np.reshape(CP, (BatchSize, 1000))
                # 筛选
                for j in range(BatchSize):
                    if TargetClass in CP[j].argsort()[-TopK:][::-1]:
                        initI[count] = temp[j]
                        count += 1
                        if count == INumber:
                            break
                print("count: ", count," StartStdDeviation: ",StartStdDeviation)

                if i == 0 and count > 4:
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

                if i == 0 and count < 5:
                    Times += 1
                    if Times == 5 :
                        StartStdDeviation += 0.01
                        DNP = ENP + StartStdDeviation
                        Times = 0




            # T,C, P= sess.run([TempImg,Confidence, Prediction])
            # C,P,I,ENP,DNP,PBF,PB = sess.run([Confidence,Prediction,IndividualFitness,Expectation,StdDeviation,PbestFitness,Pbest],feed_dict={Individual:initI})
            # we need updata E and D to updata I
            # 我们需要更新ED来更新I
            # 我们发现使用PBEST更新下一代比使用GBEST更新下一代更加具有随机性。
            initI = np.clip(initI,Downer,Upper)
            # testmax = np.max(initI)
            # testmin = np.min(initI)

            # ENP,DNP,PBF,PB = sess.run([Expectation,StdDeviation,PbestFitness,Pbest],feed_dict={Individual: initI,Confidenceplaceholder:logitnp,Predictiondsplaceholder:prednp})
            LastPBF = PBF
            ENP,DNP,PBF,PB = sess.run([Expectation,StdDeviation,PbestFitness,Pbest],feed_dict={Individual: initI})
            if PBF > GBF:
                GB = PB
                GBF = PBF
                # GENP = ENP
                # GDNP = DNP

            if GB.shape[0] > 1:
                GB = GB[0]
                DNP += VectorWeight*10
                ENP += (SourceImg - (StartImg + ENP)) * VectorWeight*10
            render_frame(sess, GB, i)

            End = time.time()
            if abs(LastPBF - PBF) < Convergence:
                DNP += VectorWeight
                ENP += (SourceImg-(StartImg+ENP))*VectorWeight

            PBL2Distance = np.sqrt(np.sum(np.square(StartImg + PB - SourceImg), axis=(1, 2, 3)))
            LogText = "Step %05d: GBF: %.4f PBF: %.4f UseingTime: %.4f GBL2Distance: %.4f" %(i,GBF,PBF,End-Start,PBL2Distance)
            # testext = "\n AvgE: {} \nAvgD: {}".format(ENP,DNP)
            LogFile.write(LogText+'\n')
            print(LogText)
            # print("AvgE: ",ENP)
            # print("AvgD: ",DNP)

            if GBF > -1e-3:
                break



def get_image(index):
    global InputDir
    data_path = os.path.join(InputDir, 'val')
    image_paths = sorted([os.path.join(data_path, i) for i in os.listdir(data_path)])
    # 修改
    # assert len(image_paths) == 50000
    labels_path = os.path.join(InputDir, 'val.txt')
    with open(labels_path) as labels_file:
        labels = [i.split(' ') for i in labels_file.read().strip().split('\n')]
        labels = {os.path.basename(i[0]): int(i[1]) for i in labels}

    def get(index):
        path = image_paths[index]
        x = load_image(path)
        y = labels[os.path.basename(path)]
        return x, y

    return get(index)

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

# def set_value_4D(matrix,i,j,k,l,val):
#     size1 = int(matrix.get_shape()[0])
#     size2 = int(matrix.get_shape()[1])
#     size3 = int(matrix.get_shape()[2])
#     size4 = int(matrix.get_shape()[3])
#     val_diff = val - matrix[i][j][k][l]
#     diff_matrix = tf.sparse_tensor_to_dense(tf.SparseTensor(indices=[i,j,k,l], values=[val_diff], dense_shape=[size1,size2,size3,size4]))
#     matrix.assign_add(diff_matrix)
#     return matrix

# 二维
# def set_value(matrix, x, y, val):
#     # 得到张量的宽和高，即第一维和第二维的Size
#     w = int(matrix.get_shape()[0])
#     h = int(matrix.get_shape()[1])
#     # 构造一个只有目标位置有值的稀疏矩阵，其值为目标值于原始值的差
#     val_diff = val - matrix[x][y]
#     diff_matrix = tf.sparse_tensor_to_dense(tf.SparseTensor(indices=[x, y], values=[val_diff], dense_shape=[w, h]))
#     # 用 Variable.assign_add 将两个矩阵相加
#     matrix.assign_add(diff_matrix)

if __name__ == '__main__':
    main()

# temp = tf.Variable(np.zeros(IndividualShape),dtype=tf.float32) # （INumber，299，299，3）
# 利用新的期望与方差生成个体
# for i in range(INumber):
#     for j in range(299):
#         for k in range(299):
#             for l in range(3):
#                 temp = tf.random_normal([1],mean=Expectation[j][k][l],stddev=StdDeviation[j][k][l],dtype=tf.float32)
#                 Individual = set_value_4D(Individual,i,j,k,l,temp)
