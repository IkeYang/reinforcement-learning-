# This is for the Reinforcement Learning Project
import numpy as np
import random
from sklearn.cluster import Kmeans


def half_moon(current):
    a = random.random()
    if a > 0.5:
        action = 1
        nextp = (current + 1) % 12
    else:
        action = -1
        nextp = (current + 11) % 12
    if nextp >= 7:
        state = 0
    else:
        state = 1
    return action, state, nextp


# find indexes of some h/hlist is list
def ind_h(hlist):
    arrh = np.array(hlist)
    arrh[:, 1:6:2] = (arrh[:, 1:6:2] + np.array([1, 1, 1])) / 2
    mul = arrh * np.array([1, 2, 4, 8, 16, 32])
    ind = mul.sum(axis=1)
    return ind


# find indexes of some t
def ind_t(tlist):
    arrt = np.array(tlist)
    mul = arrt * np.array([1, 2])
    ind = mul.sum(axis=1)
    return ind


# find g(h) of some h/hlist, g are lists
def g_h(hlist, g):
    arrh = np.array(hlist)
    gh = arrh * np.array(g)
    return gh


# find p matrix
def solve_p(train):
    p_count = np.zeros(64, 4)
    for ht in train:
        id1 = ind_h(ht[0])
        id2 = ind_t(ht[1])
        p_count[id1, id2] = p_count[id1, id2] + 1
    row_sum = p_count.sum(axis=0)
    row_sum_use = row_sum(64, 1)
    p = p_count / row_sum_use
    return p


# make_train, n is the size
def make_train(n):
    a = []
    s = []
    p = []
    train = []
    for i in range(5):
        ak, sk, pk = half_moon(11)
        a.append(ak)
        s.append(sk)
        p.append(pk)

    for j in range(n):
        ak, sk, pk = half_moon(s[-1])
        a.append(ak)
        s.append(sk)
        p.append(pk)
        h = [s[j + 5 - 5], a[j + 5 - 5], s[j + 5 - 4], a[j + 5 - 4], s[j + 5 - 3], a[j + 5 - 3]]
        t = [s[j + 5 - 2], s[j + 5 - 1]]
        ht = [h, t]
        train.append(ht)

    return train


# the h in order
def order_h():
    h_all = []
    for i in range(64):
        j = i
        h = []
        for k in range(5):
            if j >= pow(2, 5 - k):
                h.append(1)
                j = j - pow(2, 5 - k)
            else:
                h.append(0)
        if j == 1:
            h.append(1)
        else:
            h.append(0)
        h.reverse()
        h_all.append(h)
    return h_all


def order_t():
    t_all = [[0, 0], [1, 0], [0, 1], [1, 1]]
    return t_all


train = make_train(1000)
p = solve_p(train)
h_type = order_h()
g = [1, 1, 1, 1, 1, 1]

for ii in range(20):
    gh = g_h(h_type, g)
    kmeans = KMeans(n_cluster=20)
    kmeans.fit(gh)

#Author:ike yang