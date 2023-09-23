import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score,auc,roc_curve
import math


def sample_binormal_scores(Npos,POS_MU,POS_SIGMA,Nneg,NEG_MU, NEG_SIGMA):
	pos_scores = np.random.normal(POS_MU, POS_SIGMA, Npos)
	neg_scores = np.random.normal(NEG_MU, NEG_SIGMA, Nneg)
	return pos_scores,neg_scores


#1-D numpy array or list
def calc_aupr_adj(y_true, y_score):
	assert(len(y_true)==len(y_score))
	n=len(y_score)
	npos=sum(y_true)
	nneg=n-npos
	ranking=sorted([(a,b) for a, b in zip(y_true, y_score)], reverse=True, key=lambda x:x[1])
	labels=[t[0] for t in ranking]
	
	tpi=np.cumsum(labels)
	fpri=[(i+1-tpi[i])/nneg for i in range(n)]
	
	#precision adjusted
	preci=[tpi[i]/(tpi[i]+fpri[i]*npos) for i in range(n)]
	#precision not adjusted, for validation purpose
	#preci=[tpi[i]/(i+1) for i in range(n)]

	recalli=[0]*(n+1)
	for i in range(n):
		recalli[i+1]=tpi[i]/npos

	aupr=[(b-a)*c for a,b,c in zip(recalli[:n],recalli[1:],preci)]
	return np.sum(aupr)

def compute_midrank(x):
	J = np.argsort(x)
	Z = x[J]
	N = len(x)
	T = np.zeros(N, dtype=np.float)
	i = 0
	while i < N:
		j = i
		while j < N and Z[j] == Z[i]:
			j += 1
		T[i:j] = 0.5*(i + j - 1)
		i = j
	T2 = np.empty(N, dtype=np.float)
	T2[J] = T + 1
	return T2

def DeLong(y_score,m,n):
	positive_examples = y_score[:m]
	negative_examples = y_score[m:]

	tx = compute_midrank(positive_examples)
	ty = compute_midrank(negative_examples)
	tz = compute_midrank(y_score)
	v01 = (tz[:m] - tx) / n
	v10 = 1.0 - (tz[ m:] - ty) / m
	sx = np.cov(v01)
	sy = np.cov(v10)
	delongcov = sx / m + sy / n
	return delongcov,sx,sy


def metric(y_true, y_score):
	fpr, tpr, thresh = roc_curve(y_true, y_score)
	auroc=auc(fpr, tpr)
	aupr=average_precision_score(y_true, y_score)
	aupr_adj=calc_aupr_adj(y_true, y_score)
	return auroc,aupr,aupr_adj

def evaluate(y_true,y_score):
	fpr, tpr, thresh = roc_curve(y_true, y_score)
	auroc=auc(fpr, tpr)
	aupr=average_precision_score(y_true, y_score)
	aupr_adj=calc_aupr_adj(y_true, y_score)
	n=len(y_score)
	npos=sum(y_true)
	nneg=n-npos
	delongcov,sx,sy = DeLong(y_score,npos,nneg)
	return auroc, delongcov, aupr, aupr_adj, sx, sy

