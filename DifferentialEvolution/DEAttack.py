# 本程序为差分进化算法实现攻击的demo
import PIL
import numpy as np
import tensorflow as tf
from scipy.stats import norm

from inception_v3_imagenet import model, SIZE
import os
import shutil
from utils import *

InputDir = "adv_samples"
OutDir = "adv_example/"
SourceIndex = 0
TargetIndex = 1
BatchSize = 100             # 染色体个数 / 个体个数
NumClasses = 1000           # 标签种类
MaxEpoch = 1000             # 迭代上限
Reserve = 0.5               # 保留率 = 父子保留的精英量 / BestNumber
BestNmber = int(BatchSize*Reserve) # 优秀样本数量
IndividualShape = (BatchSize,299,299,3)
Directions = 299*299*3
ImageShape = (299,299,3)
def main():
    global OutDir
    global MaxEpoch

    SourceImg,SourceClass = get_image(SourceIndex)
    TargetImg, TargetClass = get_image(TargetIndex)

    if os.path.exists(OutDir):
        shutil.rmtree(OutDir)
    os.makedirs(OutDir)
    one_hot_vec = one_hot(TargetClass, NumClasses)
    with tf.Session() as sess:

        # 完成输入
        InputImg = tf.constant(SourceImg) #（299，299，3）
        TempImg = tf.expand_dims(InputImg,axis=0)   #（1，299，299，3）
        Labels = np.repeat(np.expand_dims(one_hot_vec, axis=0),repeats=BatchSize, axis=0) # （BatchSize，1000）

        # 开始进化算法
        Individual = tf.placeholder(shape=IndividualShape,dtype=tf.float32) # （BatchSize，299，299，3）
        Expectation = tf.Variable(np.random.random([299,299,3]),dtype=tf.float32) # （299，299，3）
        Deviation = tf.Variable(np.random.random([299,299,3]),dtype=tf.float32) # （299，299，3）

        NewImage = Individual + SourceImg

        # 计算置信度
        # Confidenceplaceholder = tf.placeholder(dtype=tf.float32)
        # Predictiondsplaceholder = tf.placeholder(dtype=tf.int32)
        # Confidence = Confidenceplaceholder # （BatchSize，K）
        # Prediction = Predictiondsplaceholder # （BatchSize）
        Confidence,Prediction = model(sess,Individual)

        # 计算适应度
        # IndividualFitness = -tf.reduce_sum(Labels * tf.log(Confidence), 1) #（BatchSize）
        # （BatchSize，1）还是（BatchSize） ？ 是（BatchSize）
        # reduction_indices 表示求和方向，并降维
        IndividualFitness = tf.nn.softmax_cross_entropy_with_logits()


        # 选取优秀的的前BestNmber的个体
        TopKFit,TopKFitIndx =  tf.nn.top_k(IndividualFitness,BestNmber)
        TopKIndividual = tf.gather(Individual,TopKFitIndx) # (BestNmber,299,299,3) 此处是否可以完成

        # 更新期望与方差
        Expectation = tf.reduce_mean(TopKIndividual,reduction_indices=0)
        Deviation = 0
        for i in range(BestNmber):
            Deviation += tf.square(TopKIndividual[i] - Expectation)
        Deviation /= BestNmber
        StdDeviation = tf.sqrt(Deviation)

        # 获取种群最佳（活着的，不算历史的）
        PbestFitness = tf.reduce_max(IndividualFitness)
        Pbestinds = tf.where(tf.equal(PbestFitness,IndividualFitness))
        Pbestinds = Pbestinds[:,0]
        Pbest = tf.gather(Individual,Pbestinds)

        GBF = 10000.0
        GB = np.zeros([299,299,3],dtype=float)

        init = tf.initialize_all_variables()
        sess.run(init)

        ##debug
        # initI = norm.rvs(loc=0, scale=0.1, size=IndividualShape)
        # C,P,I,N,E,D = sess.run([Confidence,Prediction,Individual,NewImage,Expectation,StdDeviation],feed_dict={Individual:initI})
        ##debug

        initI = np.zeros(IndividualShape,dtype=float)
        ENP = np.zeros(ImageShape,dtype=float)
        DNP = ENP+0.1
        for i in range(MaxEpoch):
            if i == 0 :
                initI = norm.rvs(loc=0, scale=0.1, size=IndividualShape)
            else :
                for w in range(BatchSize):
                    for j in range(299):
                        for k in range(299):
                            for l in range(3):
                                initI[w][j][k][l] = norm.rvs(loc=ENP[j][k][l], scale=DNP[j][k][l],size = 1)


            ENP,DNP,PBF,PB = sess.run([Expectation,StdDeviation,PbestFitness,Pbest],feed_dict={Individual:initI})
            if PBF < GBF:
                GB = PB
                GBF = PBF
            print("Step",i,": ",GBF)
            if GBF < 1e-5:
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

def set_value_4D(matrix,i,j,k,l,val):
    size1 = int(matrix.get_shape()[0])
    size2 = int(matrix.get_shape()[1])
    size3 = int(matrix.get_shape()[2])
    size4 = int(matrix.get_shape()[3])
    val_diff = val - matrix[i][j][k][l]
    diff_matrix = tf.sparse_tensor_to_dense(tf.SparseTensor(indices=[i,j,k,l], values=[val_diff], dense_shape=[size1,size2,size3,size4]))
    matrix.assign_add(diff_matrix)
    return matrix

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

# temp = tf.Variable(np.zeros(IndividualShape),dtype=tf.float32) # （BatchSize，299，299，3）
# 利用新的期望与方差生成个体
# for i in range(BatchSize):
#     for j in range(299):
#         for k in range(299):
#             for l in range(3):
#                 temp = tf.random_normal([1],mean=Expectation[j][k][l],stddev=StdDeviation[j][k][l],dtype=tf.float32)
#                 Individual = set_value_4D(Individual,i,j,k,l,temp)
