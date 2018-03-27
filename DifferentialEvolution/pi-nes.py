from PIL import Image
from inception_v3_imagenet import model, SIZE
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from utils import *
import json
import pdb
import os
import sys
import shutil
import time
import scipy.misc
import PIL
import TencentAPI

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from imagenet_labels import label_to_name

# Things you should definitely set:
IMAGENET_PATH = 'adv_samples'
OUT_DIR = "adv_example/"
MOMENTUM = 0.5
# Things you can play around with:
BATCH_SIZE = 100
SIGMA = 1e-3
EPSILON = 0.05
EPS_DECAY = 0.005
MIN_EPS_DECAY = 5e-5
SAMPLES_PER_DRAW = 200
LEARNING_RATE = 1e-4
SAMPLES_PER_DRAW = 1000
# 修改
K = 1000
# 修改
IMG_INDEX = 0
target_image_index = 1
#
MAX_LR = 1e-2
MIN_LR = 5e-5
# Things you probably don't want to change:
MAX_QUERIES = 4000000
num_indices = 50000
num_labels = 1000

bestloss = 10000.0
bestadv = []
bestlogit = 0.0


def main():
    out_dir = OUT_DIR
    k = K
    print('Starting partial-information attack with only top-' + str(k))
    # 修改
    # target_image_index = pseudorandom_target_image(IMG_INDEX, num_indices)

    x, y = get_image(IMG_INDEX)
    orig_class = y
    initial_img = x

    target_img = None
    target_img, target_class = get_image(target_image_index)

    global bestadv
    global bestloss
    global bestlogit
    # 修改
    print('Set target class to be original img class %d for partial-info attack' % target_class)

    sess = tf.InteractiveSession()

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    os.makedirs(out_dir)
    batch_size = min(BATCH_SIZE, SAMPLES_PER_DRAW)
    assert SAMPLES_PER_DRAW % BATCH_SIZE == 0
    one_hot_vec = one_hot(target_class, num_labels)

    x = tf.placeholder(tf.float32, initial_img.shape)
    x_t = tf.expand_dims(x, axis=0)
    gpus = [get_available_gpus()[0]]
    labels = np.repeat(np.expand_dims(one_hot_vec, axis=0),
                       repeats=batch_size, axis=0)

    grad_estimates = []
    final_losses = []
    for i, device in enumerate(gpus):
        with tf.device(device):
            print('loading on gpu %d of %d' % (i + 1, len(gpus)))
            noise_pos = tf.random_normal((batch_size // 2,) + initial_img.shape)
            noise = tf.concat([noise_pos, -noise_pos], axis=0)
            eval_points = x_t + SIGMA * noise

            logitsplaceholder = tf.placeholder(dtype=tf.float32)
            predsplaceholder = tf.placeholder(dtype=tf.int32)
            # logits 表示逻辑层预测，（40，1000）
            # preds 表示预测结果（40）
            logits, preds = logitsplaceholder, predsplaceholder
            # labels表示目标分类的标签的one_hot_vec表示(1,1000)
            # losses 表示一个图片的损失，(40,1)
            losses = -tf.reduce_sum(labels * tf.log(logits), 1)
        # vals 表示前K的置信度 （40，K）
        # inds 表示前k的置信度的编号 （40，k）会排序
        # good_inds 表示目标分类的编号 （40，2）(2是用来表示位置，应该是（1，？）、（2，？）等）
        # good_images 用来选出前K中分类中有目标分类的batch，是一个一维数组，长度为有用的batch数
        # losses 和 noise 获取有用的损失和噪音们
        vals, inds = tf.nn.top_k(logits, k=K)
        good_inds = tf.where(tf.equal(inds, tf.constant(target_class)))  # returns (# true) x 3
        good_images = good_inds[:, 0]  # inds of img in batch that worked
        losses = tf.gather(losses, good_images)
        noise = tf.gather(noise, good_images)

        # # 优化
        minlosse = tf.reduce_min(losses)
        mininds = tf.where(tf.equal(losses,minlosse))
        mininds = mininds[:,0]
        nowbest = tf.gather(eval_points,mininds)
        # nowbest = tf.reshape(nowbest,initial_img.shape)
        # #如果不在目标分类不在前5种，就会导致，所有的loss都一。

        # 将损失平铺到每一个像素
        losses_tiled = tf.tile(tf.reshape(losses, (-1, 1, 1, 1)), (1,) + initial_img.shape)
        # 论文中的公式
        grad_estimates.append(tf.reduce_mean(losses_tiled * noise,axis=0) / SIGMA)
        # 将本次循环中的有效损失都收集起来
        final_losses.append(losses)
    # N次循环后梯度的平均值
    grad_estimate = tf.reduce_mean(grad_estimates, axis=0)
    # N次循环后的全部loss
    final_losses = tf.concat(final_losses, axis=0)

    # eval network
    with tf.device(gpus[0]):
        # instant
        # eval_logits, eval_preds = model(sess, x_t)
        eval_logits = tf.placeholder(dtype=tf.float32)
        eval_preds = tf.placeholder(dtype=tf.int32)
        #
        eval_adv = tf.reduce_sum(tf.to_float(tf.equal(eval_preds, target_class)))

    samples_per_draw = SAMPLES_PER_DRAW

    def get_grad(pt, should_calc_truth=False):
        num_batches = samples_per_draw // batch_size
        losses = []
        grads = []
        global bestloss
        global bestadv
        global bestlogit
        # 测试减少循环上限。
        for _ in range(5):
            print("get_grad", _)
            # 适配修改
            NewImages = sess.run(eval_points, {x: pt})
            logitnp, predsnp = TencentAPI.Predict(NewImages)
            # 适配修改

            # loss, dl_dx_ = sess.run([final_losses, grad_estimate], \
            #     {x: pt, logitsplaceholder: logitnp, preds: predsnp})
            loss, dl_dx_,bestadvtemp,bestlosstemp,minindstemp= sess.run([final_losses, grad_estimate,nowbest,minlosse,mininds],\
                                                            {x:pt,logitsplaceholder:logitnp,preds:predsnp})

            if len(minindstemp)>1:
                bestadvtemp = bestadvtemp[0]
            if bestlosstemp < bestloss :
                bestloss = bestlosstemp
                bestadv = bestadvtemp
                bestlogit,temploss = getLogitAndLoss(bestadv)
                bestlogit = bestlogit [0][target_class]

            losses.append(np.mean(loss))
            grads.append(dl_dx_)
        return np.array(losses).mean(), np.mean(np.array(grads), axis=0)

    with tf.device('/cpu:0'):
        render_feed = tf.placeholder(tf.float32, initial_img.shape)
        render_exp = tf.expand_dims(render_feed, axis=0)

        # 重点修改
        # render_logits, _ = model(sess, render_exp)
        # 重点修改
        render_logits = tf.placeholder(dtype=tf.float32)

    def render_frame(image, save_index):
        # actually draw the figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
        # image
        ax1.imshow(image)
        fig.sca(ax1)
        plt.xticks([])
        plt.yticks([])
        # classifications

        # instant 适配修改
        TempImage = sess.run(render_exp, {render_feed: image})
        render_logitsnp, _ = TencentAPI.Predict(TempImage)
        # 适配修改

        # 删去softmax
        probs = sess.run(render_logits, {render_feed: image, render_logits: render_logitsnp})[0]
        topk = probs.argsort()[-5:][::-1]
        topprobs = probs[topk]
        barlist = ax2.bar(range(5), topprobs)
        for i, v in enumerate(topk):
            if v == orig_class:
                barlist[i].set_color('g')
            if v == target_class:
                barlist[i].set_color('r')
        plt.sca(ax2)
        plt.ylim([0, 1.1])
        plt.xticks(range(5), [label_to_name(i)[:15] for i in topk], rotation='vertical')
        fig.subplots_adjust(bottom=0.2)

        path = os.path.join(out_dir, 'frame%06d.png' % save_index)
        if os.path.exists(path):
            os.remove(path)
        plt.savefig(path)
        plt.close()

    def getLogitAndLoss(image):
        logitsnp, predsnp = TencentAPI.Predict(np.reshape(image,(1,299,299,3)))
        loss = -np.sum(labels * np.log(logitsnp))
        return logitsnp,loss


    adv = initial_img.copy()
    bestadv = adv
    assert out_dir[-1] == '/'

    log_file = open(os.path.join(out_dir, 'log.txt'), 'w+')
    g = 0
    num_queries = 0

    last_ls = []
    current_lr = LEARNING_RATE

    max_iters = int(np.ceil(MAX_QUERIES // SAMPLES_PER_DRAW))
    real_eps = 0.5

    lrs = []
    bestloss = 10000.0
    max_lr = MAX_LR
    epsilon_decay = EPS_DECAY
    last_good_adv = adv
    nowlogit = 0.0
    # 修改
    # for i in range(max_iters):
    for i in range(30):
        start = time.time()
        render_frame(adv, i)
        render_frame(bestadv,i+100)

        # see if we should stop
        # 改
        tempImage = sess.run(x_t, {x: adv})
        eval_logitsnp, eval_predsnp = TencentAPI.Predict(tempImage)
        eval_predsnp = np.array(eval_predsnp, dtype=int)
        # 改
        padv = sess.run(eval_adv, feed_dict={x: adv, eval_logits: eval_logitsnp, eval_preds: eval_predsnp})
        if (padv == 1) and (real_eps <= EPSILON):
            print('partial info early stopping at iter %d' % i)
            break

        assert target_img is not None
        # 上下限，修改不能偏离目标图像上下限
        lower = np.clip(target_img - real_eps, 0., 1.)
        upper = np.clip(target_img + real_eps, 0., 1.)
        prev_g = g
        l, g = get_grad(adv)

        if l < 0.2:
            real_eps = max(EPSILON, real_eps - epsilon_decay)
            max_lr = MAX_LR
            last_good_adv = adv
            epsilon_decay = EPS_DECAY
            if real_eps <= EPSILON:
                samples_per_draw = 5000
            last_ls = []

        # simple momentum
        # 这里使用动量策略
        g = MOMENTUM * prev_g + (1.0 - MOMENTUM) * g

        # l是平均值，早停系统
        last_ls.append(l)
        last_ls = last_ls[-5:]
        if last_ls[-1] > last_ls[0] and len(last_ls) == 5:
            if max_lr > MIN_LR:
                print("ANNEALING MAX LR")
                max_lr = max(max_lr / 2.0, MIN_LR)
            else:
                print("ANNEALING EPS DECAY")
                adv = last_good_adv  # start over with a smaller eps
                l, g = get_grad(adv)
                assert (l < 1)
                epsilon_decay = max(epsilon_decay / 2, MIN_EPS_DECAY)
            last_ls = []

        # 此处为得到新图片的位置
        # backtracking line search for optimal lr
        current_lr = max_lr
        while current_lr > MIN_LR:
            # 梯度下降
            proposed_adv = adv - current_lr * np.sign(g)
            # 修改后的图片在目标图像的上下限
            proposed_adv = np.clip(proposed_adv, lower, upper)
            num_queries += 1

            # 适配修改
            tempImage = sess.run(x_t, {x: proposed_adv})
            eval_logitsnp, eval_predsnp = TencentAPI.Predict(tempImage)
            eval_predsnp = np.array(eval_predsnp, dtype=int)
            eval_predsnp = np.array(eval_predsnp, dtype=int)
            # 适配修改

            # 获取修改后图像分类及评分
            eval_logits_ = \
            sess.run(eval_logits, {x: proposed_adv, eval_logits: eval_logitsnp, eval_preds: eval_predsnp})[0]
            # 如果目标分类在修改后的图片的分类前K种中，产生对抗，否则，减小步长，再来
            if target_class in eval_logits_.argsort()[-k:][::-1]:
                lrs.append(current_lr)
                adv = proposed_adv
                nowlogit = eval_logits_[target_class]
                if nowlogit > bestlogit :
                    bestlogit = nowlogit
                    bestadv = adv
                    templogit,bestloss = getLogitAndLoss(adv)
                break
            else:
                current_lr = current_lr / 2
                print('backtracking, lr = %.2E' % current_lr)

        num_queries += SAMPLES_PER_DRAW

        log_text = 'Step %05d: loss %.4f nowlogit %.4f bestloss %.4f bestlogit %.4f eps %.4f eps-decay %.4E lr %.2E (time %.4f)' \
                   % (i, l,nowlogit, bestloss,bestlogit,real_eps,epsilon_decay,current_lr,time.time() - start)
        log_file.write(log_text + '\n')
        print(log_text)

        np.save(os.path.join(out_dir, '%s.npy' % (i + 1)), adv)
        scipy.misc.imsave(os.path.join(out_dir, '%s.jpg' % (i + 1)), adv)
        scipy.misc.imsave(os.path.join(out_dir, '%s.jpg' % (i +100+ 1)), bestadv)


def pseudorandom_target(index, total_indices, true_class):
    rng = np.random.RandomState(index)
    target = true_class
    while target == true_class:
        target = rng.randint(0, total_indices)
    return target


def pseudorandom_target_image(orig_index, total_indices):
    rng = np.random.RandomState(orig_index)
    target_img_index = orig_index
    while target_img_index == orig_index:
        target_img_index = rng.randint(0, total_indices)
    return target_img_index


def get_image(index):
    data_path = os.path.join(IMAGENET_PATH, 'val')
    image_paths = sorted([os.path.join(data_path, i) for i in os.listdir(data_path)])
    # 修改
    # assert len(image_paths) == 50000
    labels_path = os.path.join(IMAGENET_PATH, 'val.txt')
    with open(labels_path) as labels_file:
        labels = [i.split(' ') for i in labels_file.read().strip().split('\n')]
        labels = {os.path.basename(i[0]): int(i[1]) for i in labels}

    def get(index):
        path = image_paths[index]
        x = load_image(path)
        y = labels[os.path.basename(path)]
        return x, y

    return get(index)


# get center crop
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


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


if __name__ == '__main__':
    main()
