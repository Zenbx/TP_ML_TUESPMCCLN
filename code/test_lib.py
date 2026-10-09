import numpy as np, time
from lib import *
# gradient check of EDL loss
rng=np.random.RandomState(0)
Z=rng.randn(6,3); y=rng.randint(0,3,6); Y=np.eye(3)[y]; W=rng.rand(6)+.5
l,g=edl_loss_grad(Z,Y,W,0.3)
num=np.zeros_like(Z); h=1e-6
for i in range(6):
    for j in range(3):
        Zp=Z.copy(); Zp[i,j]+=h; Zm=Z.copy(); Zm[i,j]-=h
        num[i,j]=(edl_loss_grad(Zp,Y,W,0.3)[0]-edl_loss_grad(Zm,Y,W,0.3)[0])/(2*h)
print("max grad err", np.abs(num-g).max(), np.abs(g).max())
# quick sanity on synthetic
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
X,y=make_classification(8000,40,n_informative=12,n_redundant=6,weights=[.92,.08],class_sep=.8,flip_y=.02,n_clusters_per_class=2,random_state=0)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.4,stratify=y,random_state=0)
sc=StandardScaler().fit(Xtr); Xtr,Xte=sc.transform(Xtr),sc.transform(Xte)
for name,m in [("sga",LogRegSGA()),("edl",EDLLinear()),("edl-rff",EDLLinear(rff=300))]:
    t=time.time(); m.fit(Xtr,ytr); P=m.predict_proba(Xte)
    print(name, "AUC %.3f BA %.3f ECE %.3f t=%.1fs"%(roc_auc_score(yte,P[:,1]),balanced_accuracy_score(yte,P.argmax(1)),ece_score(P,yte),time.time()-t))
from sklearn.svm import SVC
t=time.time(); s=SVC(gamma='scale').fit(Xtr[:4800],ytr[:4800]); print("svc t",time.time()-t, roc_auc_score(yte,s.decision_function(Xte)))
