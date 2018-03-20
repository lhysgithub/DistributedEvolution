# 待改进 本程序使用固定输入输出方式
# 待改进 本程序使用随机化交叉突变概率
import PIL
import numpy as np
import tensorflow as tf
import os
import shutil
from utils import *

InputDir = "adv_samples"
OutDir = "adv_example/"
SourceIndex = 0
TargetIndex = 1
BatchSize = 100             # 染色体个数
NumClasses = 1000           # 标签种类
ChromosomesShape = (BatchSize,299,299,3)
def main():
    global OutDir

    SourceImg,SourceClass = get_image(SourceIndex)
    TargetImg, TargetClass = get_image(TargetIndex)

    if os.path.exists(OutDir):
        shutil.rmtree(OutDir)
    os.makedirs(OutDir)
    one_hot_vec = one_hot(TargetClass, NumClasses)
    with tf.Session() as sess:
        InputImg = tf.placeholder(tf.float32,SourceImg.shape)
        TempImg = tf.expand_dims(InputImg,axis=0)
        Labels = np.repeat(np.expand_dims(one_hot_vec, axis=0),repeats=BatchSize, axis=0)

        # 开始进化算法
        Chromosomes = tf.placeholder(tf.float32,ChromosomesShape)

        ChromosomesPC = tf.Variable(np.random.random(BatchSize),dtype=tf.float32)
        ChromosomesPM = tf.Variable(np.random.random(BatchSize),dtype=tf.float32)

        Confidenceplaceholder = tf.placeholder(dtype=tf.float32)
        Predictiondsplaceholder = tf.placeholder(dtype=tf.int32)
        Confidence = Confidenceplaceholder
        Prediction = Predictiondsplaceholder

        ChromosomesFitness = -tf.reduce_sum(Labels * tf.log(Confidence), 1)

        # 获取种群最佳（活着的，不算历史的）
        Pbestinds = tf.where(tf.equal(tf.reduce_max(ChromosomesFitness),ChromosomesFitness))
        Pbestinds = Pbestinds[:,0]
        Pbest = tf.gather(Chromosomes,Pbestinds)

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