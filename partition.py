#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np; np.random.seed(1234)
import pandas as pd


ntrain = 150000

data = pd.read_csv('../ratings.txt', sep='\t', quoting=3)
data = pd.DataFrame(np.random.permutation(data))
trn, tst = data[:ntrain], data[ntrain:]

header = 'id document label'.split()
trn.to_csv('C:/Users/hongs/OneDrive/문서/대학교 강의 관련/Phything(학회 세미나)/Naver sentiment movie corpus/nsmc-master/ratings_train.txt', sep='\t', index=False, header=header)
tst.to_csv('C:/Users/hongs/OneDrive/문서/대학교 강의 관련/Phything(학회 세미나)/Naver sentiment movie corpus/nsmc-master/ratings_test.txt', sep='\t', index=False, header=header)
