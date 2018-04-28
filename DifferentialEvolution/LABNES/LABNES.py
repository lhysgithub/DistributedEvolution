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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from imagenet_labels import label_to_name
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Things you should definitely set:
IMAGENET_PATH = 'adv_samples/'
OUT_DIR = "adv_example/"
MOMENTUM = 0.0
# Things you can play around with:
BATCH_SIZE = 40
SIGMA = 1e-3
EPSILON = 0.05
EPS_DECAY = 0.005
MIN_EPS_DECAY = 5e-5
SAMPLES_PER_DRAW = 200
LEARNING_RATE = 1e-4
# SAMPLES_PER_DRAW = 1000
K = 5
MAX_LR = 1e-2
MIN_LR = 5e-5
# Things you probably don't want to change:
MAX_QUERIES = 4000000
num_indices = 50000
num_Labels = 1000
IMG_INDEX = 0
target_image_index = 1


def main():
    out_dir = OUT_DIR
    k = K
    print('Starting partial-information attack with only top-' + str(k))
    # target_image_index = pseudorandom_target_image(IMG_INDEX, num_indices)


    # TargetClass = OrigClass
    # print('Set target class to be original img class %d for partial-info attack' % TargetClass)

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    os.makedirs(out_dir)
    batch_size = min(BATCH_SIZE, SAMPLES_PER_DRAW)
    assert SAMPLES_PER_DRAW % BATCH_SIZE == 0

    # 以上全部移到循环
    with tf.Session() as sess:

        labels = tf.placeholder(tf.float32, shape=(batch_size,num_Labels))
        Init_Img = tf.placeholder(tf.float32, shape=(299,299,3))
        Target_Img = tf.placeholder(tf.float32, shape=(299,299,3))
        Target_Class = tf.placeholder(tf.int64)

        x = tf.placeholder(tf.float32, Init_Img.shape)
        x_t = tf.expand_dims(x, axis=0)
        # gpus = [get_available_gpus()[0]]

        grad_estimates = []
        final_losses = []
        # for i, device in enumerate(gpus):
        #     with tf.device(device):
        # print('loading on gpu %d of %d' % (i+1, len(gpus)))
        noise_pos = tf.random_normal((batch_size//2,299,299,3))
        noise = tf.concat([noise_pos, -noise_pos], axis=0)
        eval_points = x_t + SIGMA * noise
        logits, preds = model(sess, eval_points)
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        vals, inds = tf.nn.top_k(logits, k=K)
            # inds is batch_size x k
        good_inds = tf.where(tf.equal(inds, tf.cast(Target_Class,dtype=tf.int32))) # returns (# true) x 3
        good_images = good_inds[:,0] # inds of img in batch that worked
        losses = tf.gather(losses, good_images)
        noise = tf.gather(noise, good_images)

        losses_tiled = tf.tile(tf.reshape(losses, (-1, 1, 1, 1)), (1,299,299,3))
        grad_estimates.append(tf.reduce_mean(losses_tiled * noise, \
            axis=0)/SIGMA)
        final_losses.append(losses)
        grad_estimate = tf.reduce_mean(grad_estimates, axis=0)
        finallosses = tf.concat(final_losses, axis=0)

        # eval network
        # with tf.device(gpus[0]):
        eval_logits, eval_preds = model(sess, x_t)
        eval_adv = tf.reduce_sum(tf.to_float(tf.equal(eval_preds, Target_Class)))

        # with tf.device('/cpu:0'):
        render_feed = tf.placeholder(tf.float32, Init_Img.shape)
        render_exp = tf.expand_dims(render_feed, axis=0)
        render_logits, _ = model(sess, render_exp)

        samples_per_draw = SAMPLES_PER_DRAW
        def get_grad(pt,InitImg,TargetImg,TargetClass,Labels,should_calc_truth=False):
            num_batches = samples_per_draw // batch_size
            losses = []
            grads = []
            # grad_estimates = []
            # final_losses = []
            feed_dict = {x: pt,Init_Img:InitImg,Target_Img:TargetImg,Target_Class:TargetClass,labels:Labels}
            for _ in range(num_batches):
                loss, dl_dx_ = sess.run([finallosses, grad_estimate], feed_dict)
                losses.append(np.mean(loss))
                grads.append(dl_dx_)
            return np.array(losses).mean(), np.mean(np.array(grads), axis=0)



        def render_frame(image, save_index,OrigClass,TargetClass):
            # actually draw the figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
            # image
            ax1.imshow(image)
            fig.sca(ax1)
            plt.xticks([])
            plt.yticks([])
            # classifications
            probs = softmax(sess.run(render_logits, {render_feed: image})[0])
            topk = probs.argsort()[-5:][::-1]
            topprobs = probs[topk]
            barlist = ax2.bar(range(5), topprobs)
            for i, v in enumerate(topk):
                if v == OrigClass:
                    barlist[i].set_color('g')
                if v == TargetClass:
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

        for p in range(16,100):
            target_image_index = p//10
            IMG_INDEX = p%10
            if IMG_INDEX == target_image_index:
                continue
            InitImg, OrigClass = get_image(IMG_INDEX)
            TargetImg, _ = get_image(target_image_index)
            TargetClass = OrigClass
            # 最终目的，将targetimg分类为origclass
            one_hot_vec = one_hot(TargetClass, num_Labels)
            Labels = np.repeat(np.expand_dims(one_hot_vec, axis=0),
                            repeats=batch_size, axis=0)

            adv = InitImg.copy()
            assert out_dir[-1] == '/'

            log_file = open(os.path.join(out_dir, 'log%d.txt'%p), 'w+')
            g = 0
            num_queries = 0

            last_ls = []
            current_lr = LEARNING_RATE

            max_iters = int(np.ceil(MAX_QUERIES // SAMPLES_PER_DRAW))
            real_eps = 0.5
    
            lrs = []
            max_lr = MAX_LR
            epsilon_decay = EPS_DECAY
            last_good_adv = adv

            for i in range(max_iters):
                start = time.time()
                render_frame(adv, p,OrigClass,TargetClass)

                # see if we should stop

                padv = sess.run(eval_adv, feed_dict=
                {x: adv,Init_Img:InitImg,Target_Img:TargetImg,Target_Class:TargetClass,labels:Labels})
                if (padv == 1) and (real_eps <= EPSILON):
                    print('partial info early stopping at iter %d' % i)
                    break

                start1 = time.time()
                assert TargetImg is not None
                lower = np.clip(TargetImg - real_eps, 0., 1.)
                upper = np.clip(TargetImg + real_eps, 0., 1.)
                prev_g = g
                l, g = get_grad(adv,InitImg,TargetImg,TargetClass,Labels)
                num_queries += SAMPLES_PER_DRAW
                end1 = time.time()
                if l < 0.2:
                    real_eps = max(EPSILON, real_eps - epsilon_decay)
                    max_lr = MAX_LR
                    last_good_adv = adv
                    epsilon_decay = EPS_DECAY
                    if real_eps <= EPSILON:
                        samples_per_draw = 5000
                    last_ls = []

                # simple momentum
                g = MOMENTUM * prev_g + (1.0 - MOMENTUM) * g

                last_ls.append(l)
                last_ls = last_ls[-5:]
                if last_ls[-1] > last_ls[0] and len(last_ls) == 5:
                    if max_lr > MIN_LR:
                        print("ANNEALING MAX LR")
                        max_lr = max(max_lr / 2.0, MIN_LR)
                    else:
                        print("ANNEALING EPS DECAY")
                        adv = last_good_adv # start over with a smaller eps
                        l, g = get_grad(adv,InitImg,TargetImg,TargetClass,Labels)
                        num_queries += SAMPLES_PER_DRAW
                        assert (l < 1)
                        epsilon_decay = max(epsilon_decay / 2, MIN_EPS_DECAY)
                    last_ls = []

                # backtracking line search for optimal lr
                current_lr = max_lr
                while current_lr > MIN_LR:
                    proposed_adv = adv - current_lr * np.sign(g)
                    proposed_adv = np.clip(proposed_adv, lower, upper)
                    num_queries += 1
                    eval_logits_ = sess.run(eval_logits,
                                            {x: proposed_adv,Init_Img:InitImg,Target_Img:TargetImg,Target_Class:TargetClass,labels:Labels})[0]
                    if TargetClass in eval_logits_.argsort()[-k:][::-1]:
                        lrs.append(current_lr)
                        adv = proposed_adv
                        break
                    else:
                        current_lr = current_lr / 2
                        print('backtracking, lr = %.2E' % current_lr)



                log_text = 'Step %05d: loss %.4f eps %.4f eps-decay %.4E lr %.2E (time %.4f) num_queries: %d testtime %.4f p: %d' % (i, l, \
                        real_eps, epsilon_decay, current_lr, time.time() - start,num_queries,end1-start1,p)
                log_file.write(log_text + '\n')
                print(log_text)

                # np.save(os.path.join(out_dir, '%s.npy' % (i+1)), adv)
                scipy.misc.imsave(os.path.join(out_dir, '%s.png' % (p)), adv)
                # if IMG_INDEX == target_image_index:
                #     break


def pseudorandom_target(index, total_indices, true_class):
    rng = np.random.RandomState(index)
    target = true_class
    while target == true_class:
        target = rng.randint(0, total_indices)
    return target

def pseudorandom_target_image(orig_index, total_indices):
    rng = np.random.RandomState(orig_index)
    TargetImg_index = orig_index
    while TargetImg_index == orig_index:
        TargetImg_index = rng.randint(0, total_indices)
    return TargetImg_index

def get_image(index):
    data_path = os.path.join(IMAGENET_PATH, 'val')
    image_paths = sorted([os.path.join(data_path, i) for i in os.listdir(data_path)])
    # 修改
    # assert len(image_paths) == 50000
    Labels_path = os.path.join(IMAGENET_PATH, 'val.txt')
    with open(Labels_path) as Labels_file:
        Labels = [i.split(' ') for i in Labels_file.read().strip().split('\n')]
        Labels = {os.path.basename(i[0]): int(i[1]) for i in Labels}
    def get(index):
        path = image_paths[index]
        x = load_image(path)
        y = Labels[os.path.basename(path)]
        return x, y
    return get(index)

# get center crop
def load_image(path):
    image = PIL.Image.open(path)
    if image.height > image.width:
        height_off = int((image.height - image.width)/2)
        image = image.crop((0, height_off, image.width, height_off+image.width))
    elif image.width > image.height:
        width_off = int((image.width - image.height)/2)
        image = image.crop((width_off, 0, width_off+image.height, image.height))
    image = image.resize((299, 299))
    img = np.asarray(image).astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.repeat(img[:,:,np.newaxis], repeats=3, axis=2)
    if img.shape[2] == 4:
        # alpha channel
        img = img[:,:,:3]
    return img

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

if __name__ == '__main__':
    main()
