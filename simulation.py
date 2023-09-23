import pandas as pd
import numpy as np
import os
import sys
import math
import metric_utils as utils
import matplotlib.pyplot as plt
from matplotlib import colors


sample_binormal_scores=utils.sample_binormal_scores
evaluate=utils.evaluate
'''
20 steps for each phase
Phase 1: same performance, same positive ratio(binary class distribution), decreasing total number 
Phase 2: same performance, decreasing positive ratio(less positive samples), fixed total number 
Phase 3: decreased performance, same positive ratio(binary class distribution), increasing total number 
'''
TOTAL_STEP=60
CLASS_RATIO_1=0.05
CLASS_RATIO_2=0.02
N_TOTAL_1=5000
N_TOTAL_2=50
N_TOTAL_3=400
N_POS_THRESH=10
N_ts=np.concatenate([
	np.linspace(N_TOTAL_1, N_TOTAL_2, num=20, endpoint=True),
	np.repeat(N_TOTAL_3,20),
	np.linspace(N_TOTAL_3, N_TOTAL_1, num=20, endpoint=True)
	])

CLASS_RATIO_ts=np.concatenate([
	np.repeat(CLASS_RATIO_1,20),
	np.linspace(CLASS_RATIO_1, CLASS_RATIO_2, num=20, endpoint=True),
	np.repeat(CLASS_RATIO_2,20)
	])


#initial settings
#score distribution of positive samples
POS_MU=0.6
POS_SIGMA=0.3
#score distribution of negative samples
NEG_MU=0.2
NEG_SIGMA=0.3

#ground truth for phase 1 and 2
pos_scores,neg_scores = sample_binormal_scores(Npos=5000,POS_MU=POS_MU,POS_SIGMA=POS_SIGMA,Nneg=5000,NEG_MU=NEG_MU, NEG_SIGMA=NEG_SIGMA)
labels_groundtruth=[1]*len(pos_scores)+[0]*len(neg_scores)
scores_groundtruth=np.concatenate([pos_scores,neg_scores])
aucroc_groundtruth1, _, _, _, _, _ = evaluate(labels_groundtruth,scores_groundtruth)




#initialize
roc_est=0.8
KG_t=0.5
roc_diff=0.0

raw_aucroc,raw_CI_UB,raw_CI_LB=[],[],[]
filtered_aucroc,filtered_CI_UB,filtered_CI_LB=[],[],[]


for t in range(60):

	if t==40:
		#Phase 3 starts
		#score distribution of positive samples
		POS_MU=0.6
		POS_SIGMA=0.4
		#score distribution of negative samples
		NEG_MU=0.3
		NEG_SIGMA=0.4
		#ground truth for phase 3
		pos_scores,neg_scores = sample_binormal_scores(Npos=5000,POS_MU=POS_MU,POS_SIGMA=POS_SIGMA,Nneg=5000,NEG_MU=NEG_MU, NEG_SIGMA=NEG_SIGMA)
		labels_groundtruth=[1]*len(pos_scores)+[0]*len(neg_scores)
		scores_groundtruth=np.concatenate([pos_scores,neg_scores])
		aucroc_groundtruth3, _, _, _, _, _ = evaluate(labels_groundtruth,scores_groundtruth)


	n_total_t = math.floor(N_ts[t])
	npos_t = math.ceil(n_total_t*CLASS_RATIO_ts[t])
	nneg_t = n_total_t-npos_t

	#sample scores
	pos_scores,neg_scores = sample_binormal_scores(Npos=npos_t,POS_MU=POS_MU,POS_SIGMA=POS_SIGMA,Nneg=nneg_t,NEG_MU=NEG_MU, NEG_SIGMA=NEG_SIGMA)
	#calculate labels
	labels_t=[1]*len(pos_scores)+[0]*len(neg_scores)
	scores_t=np.concatenate([pos_scores,neg_scores])

	aucroc_t, delongcov_t, aucpr_t, aucpr_adj_t, sx_t, sy_t = evaluate(labels_t,scores_t)
	
	raw_aucroc.append(aucroc_t)
	raw_CI_UB.append(min(1.0,aucroc_t+1.96*delongcov_t))
	raw_CI_LB.append(max(0.0,aucroc_t-1.96*delongcov_t))

	if t==0:
		var_est=delongcov_t
		var_extrapolated=delongcov_t
		sx_est=sx_t
		sy_est=sy_t

	if npos_t<N_POS_THRESH:
		var_est=(1/npos_t)+(1/nneg_t)
	
	#filter
	var_extrapolated=(1/npos_t)*sx_est+(1/nneg_t)*sy_est
	KG_t=var_extrapolated/(var_extrapolated+delongcov_t)
	roc_diff=aucroc_t-roc_est
	roc_est=roc_est+KG_t*roc_diff
	var_est=(1-KG_t)*var_extrapolated
	sx_est=(1-KG_t)*sx_t
	sy_est=(1-KG_t)*sy_t

	filtered_aucroc.append(roc_est)
	filtered_CI_UB.append(min(1.0,roc_est+1.96*var_est))
	filtered_CI_LB.append(max(0,roc_est-1.96*var_est))

ground_truth = np.array([aucroc_groundtruth1]*40 + [aucroc_groundtruth3]*20)

sim_data={
	"time_step":np.arange(60),
	"ground_truth_AUCROC":ground_truth,
	"raw_AUCROC":raw_aucroc,
	"raw_AUCROC_UB":raw_CI_UB,
	"raw_AUCROC_LB":raw_CI_LB,
	"filtered_AUCROC":filtered_aucroc,
	"filtered_AUCROC_UB":filtered_CI_UB,
	"filtered_AUCROC_LB":filtered_CI_LB,
}
sim_data=pd.DataFrame(data=sim_data)
sim_data.to_csv(os.path.join('res','similation_result.csv'),index=False)


plt.figure()
plt.plot(np.arange(60),raw_aucroc,label='raw',color='b')
plt.plot(np.arange(60),ground_truth,label='ground truth',color='k')
plt.plot(np.arange(60),filtered_aucroc,label='filtered',color='r')
plt.legend(loc='best')
plt.savefig('simres.png')
plt.close()



