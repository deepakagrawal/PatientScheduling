from __future__ import division
import os
import numpy as np
import itertools
import tempfile
from gams import *
from scipy.stats import poisson
import scipy.optimize as opt
import shutil


class JointOptimizer(object):
    ws = None

    def __init__(self, keepfiles=False):
        JointOptimizer.tmpdir = tempfile.mkdtemp()
        JointOptimizer.ws = GamsWorkspace(debug=True if keepfiles else False,
                                           working_directory=JointOptimizer.tmpdir)

    def del_tmpdir(self):
        shutil.rmtree(JointOptimizer.tmpdir)

    @staticmethod
    def get_model_text():
        return '''
$title Sample.gms    
Set      dj     
         kj
         tj     
         cj     
         lj     
         nj     
         i     /0*100/
;
alias    (i,fi), (i,c0), (dj,di), (dj,dx), (lj,li);


Scalar
         dd
         theta    
         CAP
         Policy
         cSum
;


Parameter
         v(dj,kj,tj,cj,lj)
         r(dj)
         s(dj)
         rt(dj,dj)
         st(dj,dj)
         csign(kj,lj)
         L(cj,lj,nj)
         prob(nj)
         y(kj,cj,lj,tj,dj,dj)
         z(dj,kj,tj,cj,lj)
         loc_dep
         parami(i)
;

$if not set gdxincname $abort 'no include file name for data file provided'
$gdxin %gdxincname%
$load dj,kj,tj,cj,lj,nj,dd,theta,CAP,cSum,v,r,s,csign,L,prob, Policy,y,z, loc_dep
$gdxin


rt(dj,di)$(ord(di) le ord(dj)) = max(0.6,1 - 0.04*(ord(dj) - ord(di)));
st(dj,di)$(ord(di) le ord(dj)) = rt(dj,di);
parami(i) = ord(i) - 1;

Variables
x(dj,kj,tj,cj,lj)  scheduling probability
u(cj,lj)        leaving probability
obj               total profit
cb(kj,lj,i)     binary capacity
C(kj,lj)        capacity
y1(kj,lj)
z1(kj,lj)
z2(kj,lj)
alphaS(kj,lj,nj)
P1S(kj,lj,nj)
P2S(kj,lj,nj)
P3S(kj,lj,nj)
P4S(kj,lj,nj)
alphaD(dj,kj,lj,nj)
P1D(dj,kj,lj,nj)
P2D(dj,kj,lj,nj)
P3D(dj,kj,lj,nj)
P4D(dj,kj,lj,nj)
;


Positive Variable x,u,C,alphaS,alphaD,P3S,P4S,P1D, P2D, P3D,P4D;
Binary Variable cb;
x.up(dj,kj,tj,cj,lj) = sign(v(dj,kj,tj,cj,lj));;
u.up(cj,lj) = 1;
u.lo(cj,lj) = 1 - sign(sum((dj,kj,tj),sign(v(dj,kj,tj,cj,lj))));
cb.up(kj,lj,i) = csign(kj,lj);
*C.lo(kj,lj) = 2*csign(kj,lj);
P1S.up(kj,lj,nj) = 1;
P2S.up(kj,lj,nj) = 1;
P3S.up(kj,lj,nj) = 1;
P4S.up(kj,lj,nj) = 1;
P1D.up(dj,kj,lj,nj) = 1;
P2D.up(dj,kj,lj,nj) = 1;
P3D.up(dj,kj,lj,nj) = 1;
P4D.up(dj,kj,lj,nj) = 1;
*C.l(kj,lj) = CAP*csign(kj,lj);


*display u.lo, x.up;

Equations
total_profit_SP        Objective value SP
total_profit_DP        Objective value SP
eq1(kj,cj,lj)             Sum of probability 1 for dedicated class
eq2_loc_dep_1(cj)             Sum of probability 1 for flexible class when loc_dep = 1
eq2_loc_dep_0(cj,lj)             Sum of probability 1 for flexible class when loc_dep = 0
eq3(dj,tj,cj,lj)       Sum of probability 1 for urgent class
eq3_1(tj,cj,lj)        Sum of probability 1 for urgent class
eq4(dj,kj,tj,cj,lj) choice constraint
eq4_1(dj,kj,tj,cj,lj)
capCon1(kj,lj,i)    c_i > c_i+1
capCon2(kj,lj)      capacity constraint at certain location
capCon4
alphaEqS(kj,lj,nj)
P1eqS(kj,lj,nj)
P2eqS(kj,lj,nj)
P3eqS(kj,lj,nj)
P4eqS(kj,lj,nj)
alphaEqD(dj,kj,lj,nj)
P1eqD(dj,kj,lj,nj)
P2eqD(dj,kj,lj,nj)
P3eqD(dj,kj,lj,nj)
P4eqD(dj,kj,lj,nj)
;



total_profit_SP         .. obj =e= sum((kj,lj,nj), prob(nj)*alphaS(kj,lj,nj)) - 
                                   theta*sum((kj,lj,nj),prob(nj)*(alphaS(kj,lj,nj)*P1S(kj,lj,nj) - C(kj,lj)*P2S(kj,lj,nj) + C(kj,lj)*P3S(kj,lj,nj) - alphaS(kj,lj,nj)*P4S(kj,lj,nj)));
total_profit_DP         .. obj =e= sum((dj,kj,lj,nj), alphaD(dj,kj,lj,nj)*prob(nj))- 
                                   theta*sum((dj,kj,lj,nj),prob(nj)*(alphaD(dj,kj,lj,nj))*P1D(dj,kj,lj,nj) - C(kj,lj)*P2D(dj,kj,lj,nj) + C(kj,lj)*P3D(dj,kj,lj,nj) - alphaD(dj,kj,lj,nj)*P4D(dj,kj,lj,nj));
eq1(kj,cj,lj)$(ord(cj) >= 3 and ord(kj) eq ord(cj)-2)      .. u(cj,lj) + sum((dj,tj), x(dj,kj,tj,cj,lj)) =e= 1;
eq2_loc_dep_1(cj)$(ord(cj) eq 2)      .. sum(lj, u(cj,lj) + sum((kj,dj,tj),x(dj,kj,tj,cj,lj))) =e= 1;
eq3(dj,tj,cj,lj)$(ord(cj) eq 1 and ord(dj) eq 1)   .. sum(kj, x(dj,kj,tj,cj,lj)) =e= 1;
eq3_1(tj,cj,lj)$(ord(cj) eq 1) .. sum((kj,dj)$(ord(dj) >1), x(dj,kj,tj,cj,lj)) + u(cj,lj) =e= 0;
eq4(dj,kj,tj,cj,lj)$(ord(cj) eq 2)  .. x(dj,kj,tj,cj,lj)-v(dj,kj,tj,cj,lj)*sum(li$(ord(lj) eq ord(li)),u(cj,lj)) =l= 0;
eq4_1(dj,kj,tj,cj,lj)$(ord(cj) > 2 and ord(kj) eq ord(cj) -2)  .. x(dj,kj,tj,cj,lj)-v(dj,kj,tj,cj,lj)*u(cj,lj) =l= 0;
capCon1(kj,lj,i)$(ord(i)<=CAP+1)  .. cb(kj,lj,i) =g= cb(kj,lj,i+1);
capCon2(kj,lj)                 .. C(kj,lj) =e= sum(i$(ord(i)<=CAP+1),cb(kj,lj,i));
capCon4         .. sum((kj,lj),C(kj,lj)) =l= 50;
*********** Static
alphaEqS(kj,lj,nj)   .. alphaS(kj,lj,nj) =e= sum((dj,cj,tj), L(cj,lj,nj)*r(dj)*x(dj,kj,tj,cj,lj));
P1eqS(kj,lj,nj)      .. P1S(kj,lj,nj) =e= 1 - exp(-alphaS(kj,lj,nj))*(sum(i$(ord(i)<=CAP), cb(kj,lj,i+1)*alphaS(kj,lj,nj)**(ord(i)-1)/gamma(ord(i)) ));
P2eqS(kj,lj,nj)      .. P2S(kj,lj,nj) =e= 1 - exp(-alphaS(kj,lj,nj))*(sum(i$(ord(i)<=CAP+1), cb(kj,lj,i)*alphaS(kj,lj,nj)**(ord(i)-1)/gamma(ord(i)) ));
P3eqS(kj,lj,nj)      .. P3S(kj,lj,nj) =e= exp(-alphaS(kj,lj,nj))*(sum(i$(ord(i)<=CAP), cb(kj,lj,i+1)*alphaS(kj,lj,nj)**(ord(i)-1)/gamma(ord(i)) ));
P4eqS(kj,lj,nj)      .. P4S(kj,lj,nj) =e= exp(-alphaS(kj,lj,nj))*(sum(i$(ord(i)<=CAP-1), cb(kj,lj,i+2)*alphaS(kj,lj,nj)**(ord(i)-1)/gamma(ord(i)) ));

*********** Dynamic
alphaEqD(dj,kj,lj,nj)   .. alphaD(dj,kj,lj,nj) =e= sum((cj,di,dx)$((ord(dx) eq ord(di)+ord(dj)-1) and (ord(di) ge 1) and (ord(di) le dd +1 - ord(dj))),sum(tj,y(kj,cj,lj,tj,di,dx))*rt(dx,di)) + 
sum(cj,L(cj,lj,nj)*r(dj)*sum(tj,x(dj,kj,tj,cj,lj))) + sum((cj,di)$((ord(di) ge 1) and (ord(di) le ord(dj)-1)),L(cj,lj,nj)*r(di)*sum(tj,z(di,kj,tj,cj,lj)));
P1eqD(dj,kj,lj,nj)      .. P1D(dj,kj,lj,nj) =e= 1 - exp(-alphaD(dj,kj,lj,nj))*(sum(i$(ord(i)<=CAP), cb(kj,lj,i+1)*alphaD(dj,kj,lj,nj)**(ord(i)-1)/gamma(ord(i)) ));
P2eqD(dj,kj,lj,nj)      .. P2D(dj,kj,lj,nj) =e= 1 - exp(-alphaD(dj,kj,lj,nj))*(sum(i$(ord(i)<=CAP+1), cb(kj,lj,i)*alphaD(dj,kj,lj,nj)**(ord(i)-1)/gamma(ord(i)) ));
P3eqD(dj,kj,lj,nj)      .. P3D(dj,kj,lj,nj) =e= exp(-alphaD(dj,kj,lj,nj))*(sum(i$(ord(i)<=CAP), cb(kj,lj,i+1)*alphaD(dj,kj,lj,nj)**(ord(i)-1)/gamma(ord(i)) ));
P4eqD(dj,kj,lj,nj)      .. P4D(dj,kj,lj,nj) =e= exp(-alphaD(dj,kj,lj,nj))*(sum(i$(ord(i)<=CAP-1), cb(kj,lj,i+2)*alphaD(dj,kj,lj,nj)**(ord(i)-1)/gamma(ord(i)) ));


MODEL SP_loc_dep /eq1,eq2_loc_dep_1,eq3,eq3_1,eq4,eq4_1,capCon1,capCon2,alphaEqS,P1eqS,P2eqS,P3eqS,P4eqS, total_profit_SP/;
MODEL DP_loc_dep /eq1,eq2_loc_dep_1,eq3,eq3_1,eq4,eq4_1,capCon1,capCon2,alphaEqD,P1eqD,P2eqD,P3eqD,P4eqD, total_profit_DP/;

option MINLP = SBB;
option limrow = 0;
option limcol = 0;
option solprint = off;
*option nodlim = 5000;
*option modeltype=convert;
option optcr = 0;
DP_loc_dep.nodlim = 6000;
solve DP_loc_dep using minlp maximizing obj;

*display gbdlog;    
    '''

    def solve(self, data, dynamic, schedule,loc_dep_inp):
        db = JointOptimizer.ws.add_database()

        # Sets
        dj = db.add_set("dj", 1, "DAYS");
        [dj.add_record(str(p)) for p in range(1, data.dd + 1)]
        kj = db.add_set("kj", 1, "Physicians");
        [kj.add_record(str(p)) for p in range(1, data.dk + 1)]
        tj = db.add_set("tj", 1, "Physicians")
        [tj.add_record(str(p)) for p in range(1, data.dt + 1)]
        cj = db.add_set("cj", 1, "Class");
        [cj.add_record(str(p)) for p in range(1, data.dc + 1)]
        lj = db.add_set("lj", 1, "Location");
        [lj.add_record(str(p)) for p in range(1, data.dl + 1)]
        nj = db.add_set("nj", 1, "setProb");
        [nj.add_record(str(p)) for p in range(1, data.lprob + 1)]
        loc_dep = db.add_parameter('loc_dep', 0, "Location Dependence")

        # Parameters
        dd = db.add_parameter('dd', 0, "Max Days")
        dd.add_record().value = data.dd
        theta = db.add_parameter("theta", 0, "Overtime Unit Cost")
        theta.add_record().value = data.theta
        cap = db.add_parameter('CAP', 0, "Max Capacity")
        cap.add_record().value = data.CAP
        policy = db.add_parameter('Policy', 0, "Policy Type")
        csum = db.add_parameter('cSum', 0, "Sum of capacity")
        csum.add_record().value = data.C.sum()
        loc_dep.add_record().value = loc_dep_inp

        policy.add_record().value = 0 if dynamic == "SP" else 1 if dynamic == "SPP" else 2 if (
                dynamic in ["DP", "DCPP"]) else 3 if (dynamic == "DPP") else 4
        v = db.add_parameter_dc("v", [dj, kj, tj, cj, lj], "Choice")
        r = db.add_parameter_dc('r', [dj], "retaining prob")
        s = db.add_parameter_dc('s', [dj], "show up prob")
        L = db.add_parameter_dc('L', [cj, lj, nj], "arrival rate")
        csign = db.add_parameter_dc('csign', [kj, lj], "Capacity Constraint Parameter")
        prob = db.add_parameter_dc('prob', [nj], "doubly stochastic prob")
        y = db.add_parameter_dc('y', [kj, cj, lj, tj, dj, dj], 'Current Schedule')
        z = db.add_parameter_dc('z', [dj, kj,tj, cj, lj], 'Static Prob')

        for key, val in zip(range(1, data.dd + 1), data.r):
            r.add_record(str(key)).value = val
            s.add_record(str(key)).value = val
        for key, val in np.ndenumerate(data.v):
            v.add_record([str(i + 1) for i in key]).value = val
        for key, val in np.ndenumerate(data.l):
            L.add_record([str(i + 1) for i in key]).value = val
        for key, val in np.ndenumerate(data.csign):
            csign.add_record([str(i + 1) for i in key]).value = val
        for key, val in np.ndenumerate(data.probl):
            prob.add_record([str(i + 1) for i in key]).value = val
        for key, val in np.ndenumerate(schedule):
            y.add_record([str(i + 1) for i in key]).value = val
        for key, val in np.ndenumerate(data.Prob):
            z.add_record([str(i + 1) for i in key]).value = val

        t1 = JointOptimizer.ws.add_job_from_string(JointOptimizer.get_model_text())
        opt = JointOptimizer.ws.add_options()

        opt.defines["gdxincname"] = db.name
        # opt.all_model_types = "SBB"

        t1.run(opt, databases=db)
        return t1.out_db["x"], t1.out_db["C"]


class SingleOptimizer(object):
    ws = None

    def __init__(self, keepfiles=False):
        SingleOptimizer.tmpdir = tempfile.mkdtemp()
        SingleOptimizer.ws = GamsWorkspace(debug=True if keepfiles else False,
                                           working_directory=SingleOptimizer.tmpdir)

    def del_tmpdir(self):
        shutil.rmtree(SingleOptimizer.tmpdir)

    @staticmethod
    def get_model_text():
        return '''

$title Sample.gms    
Set      dj     
         kj
         tj
         cj     
         lj     
         nj     
         i     /0*100/
;
alias    (i,fi), (i,c0), (dj,di), (dj,dx), (lj,li);


Scalar
         dd
         theta    
         CAP
         Policy
;


Parameter
         v(dj,kj,tj,cj,lj)
         r(dj)
         s(dj)
         rt(dj,dj)
         st(dj,dj) 
         C(kj,lj)
         csign(kj,lj)
         L(cj,lj,nj)
         prob(nj)
         y(kj,cj,lj,tj,dj,dj)
         z(dj,kj,tj,cj,lj)
         loc_dep
         parami(i)
;

$if not set gdxincname $abort 'no include file name for data file provided'
$gdxin %gdxincname%
$load dj,kj,tj,cj,lj,nj,dd,theta,CAP,v,r,s,C,csign,L,prob, Policy,y,z, loc_dep
$gdxin

rt(dj,di)$(ord(di) le ord(dj)) = max(0.6,1 - 0.04*(ord(dj) - ord(di)));
st(dj,di)$(ord(di) le ord(dj)) = rt(dj,di);
parami(i) = ord(i) - 1;

Variables
x(dj,kj,tj,cj,lj)  scheduling probability
u(cj,lj)        leaving probability
obj               total profit
alphaS(kj,lj,nj)    
P1S(kj,lj,nj)
P2S(kj,lj,nj)
P3S(kj,lj,nj)
P4S(kj,lj,nj)
alphaD(dj,kj,lj,nj)    
P1D(dj,kj,lj,nj)
P2D(dj,kj,lj,nj)
P3D(dj,kj,lj,nj)
P4D(dj,kj,lj,nj)
;


Positive Variable x,u, P1S, P2S, P3S, P4S, alphaS, alphaD, P1D, P2D, P3D, P4D;
x.up(dj,kj,tj,cj,lj) = sign(v(dj,kj,tj,cj,lj));
u.up(cj,lj) = 1;
P1S.up(kj,lj,nj) = 1;
P2S.up(kj,lj,nj) = 1;
P3S.up(kj,lj,nj) = 1;
P4S.up(kj,lj,nj) = 1;




Equations
total_profit_SP        Objective value SP
total_profit_DP        Objective value SP
eq1(kj,cj,lj)             Sum of probability 1 for dedicated class
eq2_loc_dep_1(cj)             Sum of probability 1 for flexible class when loc_dep = 1
eq2_loc_dep_0(cj,lj)             Sum of probability 1 for flexible class when loc_dep = 0
eq3(dj,tj,cj,lj)       Sum of probability 1 for urgent class
eq3_1(tj,cj,lj)        Sum of probability 1 for urgent class
eq4(dj,kj,tj,cj,lj) choice constraint
eq4_1(dj,kj,tj,cj,lj)
alphaEqS(kj,lj,nj)
P1eqS(kj,lj,nj)
P2eqS(kj,lj,nj)
P3eqS(kj,lj,nj)
P4eqS(kj,lj,nj)
alphaEqD(dj,kj,lj,nj)
P1eqD(dj,kj,lj,nj)
P2eqD(dj,kj,lj,nj)
P3eqD(dj,kj,lj,nj)
P4eqD(dj,kj,lj,nj)
;

total_profit_SP         .. obj =e= sum((kj,lj,nj), prob(nj)*alphaS(kj,lj,nj)) - 
                                   theta*sum((kj,lj,nj),prob(nj)*(alphaS(kj,lj,nj)*P1S(kj,lj,nj) - C(kj,lj)*P2S(kj,lj,nj) + C(kj,lj)*P3S(kj,lj,nj) - alphaS(kj,lj,nj)*P4S(kj,lj,nj)));
total_profit_DP         .. obj =e= sum((dj,kj,lj,nj), alphaD(dj,kj,lj,nj)*prob(nj))-
                                   theta*sum((dj,kj,lj,nj),prob(nj)*(alphaD(dj,kj,lj,nj))*P1D(dj,kj,lj,nj) - C(kj,lj)*P2D(dj,kj,lj,nj) + C(kj,lj)*P3D(dj,kj,lj,nj) - alphaD(dj,kj,lj,nj)*P4D(dj,kj,lj,nj));
eq1(kj,cj,lj)$(ord(cj) >= 3 and ord(kj) eq ord(cj)-2)      .. u(cj,lj) + sum((dj,tj), x(dj,kj,tj,cj,lj)) =e= 1;
eq2_loc_dep_1(cj)$(ord(cj) eq 2)      .. sum(lj, u(cj,lj) + sum((kj,dj,tj),x(dj,kj,tj,cj,lj))) =e= 1;
eq3(dj,tj,cj,lj)$(ord(cj) eq 1 and ord(dj) eq 1)   .. sum(kj, x(dj,kj,tj,cj,lj)) =e= 1;
eq3_1(tj,cj,lj)$(ord(cj) eq 1) .. sum((kj,dj)$(ord(dj) >1), x(dj,kj,tj,cj,lj)) + u(cj,lj) =e= 0;
eq4(dj,kj,tj,cj,lj)$(ord(cj) eq 2)  .. x(dj,kj,tj,cj,lj)-v(dj,kj,tj,cj,lj)*sum(li$(ord(lj) eq ord(li)),u(cj,lj)) =l= 0;
eq4_1(dj,kj,tj,cj,lj)$(ord(cj) > 2 and ord(kj) eq ord(cj) -2)  .. x(dj,kj,tj,cj,lj)-v(dj,kj,tj,cj,lj)*u(cj,lj) =l= 0;
*********** Static
alphaEqS(kj,lj,nj)   .. alphaS(kj,lj,nj) =e= sum((dj,cj,tj), L(cj,lj,nj)*r(dj)*x(dj,kj,tj,cj,lj));
P1eqS(kj,lj,nj)      .. P1S(kj,lj,nj) =e= 1 - exp(-alphaS(kj,lj,nj))*(sum(i$(ord(i)<=floor(C(kj,lj))), alphaS(kj,lj,nj)**(ord(i)-1)/gamma(ord(i)) ));
P2eqS(kj,lj,nj)      .. P2S(kj,lj,nj) =e= 1 - exp(-alphaS(kj,lj,nj))*(sum(i$(ord(i)<=floor(C(kj,lj))+sign(C(kj,lj))), alphaS(kj,lj,nj)**(ord(i)-1)/gamma(ord(i)) ));
P3eqS(kj,lj,nj)      .. P3S(kj,lj,nj) =e= exp(-alphaS(kj,lj,nj))*(sum(i$(ord(i)<=ceil(C(kj,lj))), alphaS(kj,lj,nj)**(ord(i)-1)/gamma(ord(i)) ));
P4eqS(kj,lj,nj)      .. P4S(kj,lj,nj) =e= exp(-alphaS(kj,lj,nj))*(sum(i$(ord(i)<=ceil(C(kj,lj))-sign(C(kj,lj))), alphaS(kj,lj,nj)**(ord(i)-1)/gamma(ord(i)) ));
*********** Dynamic
alphaEqD(dj,kj,lj,nj)   .. alphaD(dj,kj,lj,nj) =e= sum((cj,di,dx)$((ord(dx) eq ord(di)+ord(dj)-1) and (ord(di) ge 1) and (ord(di) le dd +1 - ord(dj))),sum(tj,y(kj,cj,lj,tj,di,dx))*rt(dx,di)) + 
sum(cj,L(cj,lj,nj)*r(dj)*sum(tj,x(dj,kj,tj,cj,lj))) + sum((cj,di)$((ord(di) ge 1) and (ord(di) le ord(dj)-1)),L(cj,lj,nj)*r(di)*sum(tj,z(di,kj,tj,cj,lj)));
P1eqD(dj,kj,lj,nj)      .. P1D(dj,kj,lj,nj) =e= 1 - exp(-alphaD(dj,kj,lj,nj))*(sum(i$(ord(i)<=floor(C(kj,lj))), alphaD(dj,kj,lj,nj)**(ord(i)-1)/gamma(ord(i)) ));
P2eqD(dj,kj,lj,nj)      .. P2D(dj,kj,lj,nj) =e= 1 - exp(-alphaD(dj,kj,lj,nj))*(sum(i$(ord(i)<=floor(C(kj,lj))+1), alphaD(dj,kj,lj,nj)**(ord(i)-1)/gamma(ord(i)) ));
P3eqD(dj,kj,lj,nj)      .. P3D(dj,kj,lj,nj) =e= exp(-alphaD(dj,kj,lj,nj))*(sum(i$(ord(i)<=ceil(C(kj,lj))-1), alphaD(dj,kj,lj,nj)**(ord(i)-1)/gamma(ord(i)) ));
P4eqD(dj,kj,lj,nj)      .. P4D(dj,kj,lj,nj) =e= exp(-alphaD(dj,kj,lj,nj))*(sum(i$(ord(i)<=ceil(C(kj,lj))-2), alphaD(dj,kj,lj,nj)**(ord(i)-1)/gamma(ord(i)) ));


MODEL SP_loc_dep /eq1,eq2_loc_dep_1,eq3,eq3_1,eq4,eq4_1,alphaEqS,P1eqS,P2eqS,P3eqS,P4eqS, total_profit_SP/;
MODEL DP_loc_dep /eq1,eq2_loc_dep_1,eq3,eq3_1,eq4,eq4_1,alphaEqD,P1eqD,P2eqD,P3eqD,P4eqD, total_profit_DP/;

option 
   NLP = conopt;


If (Policy eq 0,
        Solve SP_loc_dep using nlp maximizing obj;
Elseif Policy eq 2,
        Solve DP_loc_dep using nlp maximizing obj;
    );

*option MINLP = SBB;
*Display x.l;  
    
    '''

    def solve(self, data, dynamic, schedule, loc_dep_inp):
        db = SingleOptimizer.ws.add_database()

        # Sets
        dj = db.add_set("dj", 1, "DAYS")
        [dj.add_record(str(p)) for p in range(1, data.dd + 1)]
        kj = db.add_set("kj", 1, "Physicians")
        [kj.add_record(str(p)) for p in range(1, data.dk + 1)]
        tj = db.add_set("tj", 1, "Physicians")
        [tj.add_record(str(p)) for p in range(1, data.dt + 1)]
        cj = db.add_set("cj", 1, "Class")
        [cj.add_record(str(p)) for p in range(1, data.dc + 1)]
        lj = db.add_set("lj", 1, "Location")
        [lj.add_record(str(p)) for p in range(1, data.dl + 1)]
        nj = db.add_set("nj", 1, "setProb")
        [nj.add_record(str(p)) for p in range(1, data.lprob + 1)]
        # Parameters
        dd = db.add_parameter('dd', 0, "Max Days")
        dd.add_record().value = data.dd
        theta = db.add_parameter("theta", 0, "Overtime Unit Cost")
        theta.add_record().value = data.theta
        cap = db.add_parameter('CAP', 0, "Max Capacity")
        cap.add_record().value = data.CAP
        policy = db.add_parameter('Policy', 0, "Policy Type")
        loc_dep = db.add_parameter('loc_dep', 0, "Location Dependence")

        policy.add_record().value = 0 if dynamic == "SP" else 1 if dynamic == "SPP" else 2 if (
                dynamic in ["DP", "DCPP"]) else 3 if (dynamic == "DPP") else 4
        loc_dep.add_record().value = loc_dep_inp
        v = db.add_parameter_dc("v", [dj, kj, tj, cj, lj], "Choice")
        r = db.add_parameter_dc('r', [dj], "retaining prob")
        s = db.add_parameter_dc('s', [dj], "show up prob")
        L = db.add_parameter_dc('L', [cj, lj, nj], "arrival rate")
        C = db.add_parameter_dc('C', [kj, lj], "Capacity Constraint Parameter")
        csign = db.add_parameter_dc('csign', [kj, lj], "Capacity Constraint Parameter")
        prob = db.add_parameter_dc('prob', [nj], "doubly stochastic prob")
        y = db.add_parameter_dc('y', [kj, cj, lj, tj, dj, dj], 'Current Schedule')
        z = db.add_parameter_dc('z', [dj, kj, tj, cj, lj], 'Static Prob')

        for key, val in zip(range(1, data.dd + 1), data.r):
            r.add_record(str(key)).value = val
            s.add_record(str(key)).value = val
        for key, val in np.ndenumerate(data.v):
            v.add_record([str(i + 1) for i in key]).value = val
        for key, val in np.ndenumerate(data.l):
            L.add_record([str(i + 1) for i in key]).value = val
        for key, val in np.ndenumerate(data.C):
            C.add_record([str(i + 1) for i in key]).value = val
        for key, val in np.ndenumerate(data.csign):
            csign.add_record([str(i + 1) for i in key]).value = val
        for key, val in np.ndenumerate(data.probl):
            prob.add_record([str(i + 1) for i in key]).value = val
        for key, val in np.ndenumerate(schedule):
            y.add_record([str(i + 1) for i in key]).value = val
        for key, val in np.ndenumerate(data.Prob):
            z.add_record([str(i + 1) for i in key]).value = val

        t1 = SingleOptimizer.ws.add_job_from_string(SingleOptimizer.get_model_text())
        opt = SingleOptimizer.ws.add_options()

        opt.defines["gdxincname"] = db.name
        # opt.all_model_types = "SBB" if data.joint == 1 else "CONOPT"
        policy.first_record()
        t1.run(opt, databases=db)

        # print(
        # "x(" + rec.key(0) + "," + rec.key(1) + + rec.key(2)+ rec.key(3) +"): level=" + str(rec.level) + " marginal=" + str(rec.marginal))
        return t1.out_db["x"], 0


class SingleProblem(object):

    @staticmethod
    def SP_var(x, u, data):
        alpha = np.einsum('dcln,dkcl->kln', np.einsum('cln,d->dcln', data.l, data.r), x)
        P1 = np.array([1 - poisson.cdf(data.C, alpha[:, :, nj]) for nj in range(data.lprob)])
        P2 = np.array([1 - poisson.cdf(data.C + np.sign(data.C), alpha[:, :, nj]) for nj in range(data.lprob)])
        P3 = np.array([poisson.cdf(data.C - np.sign(data.C), alpha[:, :, nj]) for nj in range(data.lprob)])
        P4 = np.array([poisson.cdf(data.C - 2 * np.sign(data.C), alpha[:, :, nj]) for nj in range(data.lprob)])
        return alpha, P1, P2, P3, P4

    @staticmethod
    def DP_obj(x, u, data):
        return 0

    @staticmethod
    def SP_obj(x, u, data):
        alpha, P1, P2, P3, P4 = SingleProblem.SP_var(x, u, data)
        obj = -np.einsum('kln,n->', alpha, data.probl) + data.theta * np.einsum('kln,n->', (
                np.einsum('kln,nkl->kln', alpha, P1) - np.einsum('kl,nkl->kln', data.C, P2) + np.einsum('kl,nkl->kln',
                                                                                                        data.C,
                                                                                                        P3) - np.einsum(
            'kln,nkl->kln', alpha, P4)), data.probl)
        return obj

    @staticmethod
    def SP_jacobian(x, u, data):
        alpha, P1, _, P3, _ = SingleProblem.SP_var(x, u, data)
        return np.append((-np.einsum('dcl,kl->dkcl',
                                     np.einsum('dcln,n->dcl', np.einsum('cln,d->dcln', data.l, data.r), data.probl),
                                     np.sign(data.C)) +
                          data.theta * np.einsum('dkcln,n->dkcl', np.einsum('nkl,dcln->dkcln', P1 - P3,
                                                                            np.einsum('cln,d->dcln', data.l, data.r)),
                                                 data.probl)).flatten(),
                         np.zeros(data.dc * data.dl))

    # @staticmethod
    # def SP_hessian(x,u,data):
    #     alpha = np.einsum('dcln,dkcl->kln', np.einsum('cln,d->dcln', data.l, data.r), x)
    #     return (2*data.theta * np.einsum('dkcln,n->dkcl',np.einsum('nkl,dcln->dkcln',np.array([poisson.pmf(data.C - np.sign(data.C),alpha[:,:,nj]) for nj in range(data.lprob)]),np.square(np.einsum('cln,d->dcln',data.l,data.r))),data.probl)).flatten()

    @staticmethod
    def constraint_eq(x, u, data):
        cons = list()
        for cj in range(data.dc):
            if cj == 0:
                cons = cons + [np.dot(data.csign[:, lj], x[0, :, cj, lj]) - 1 for lj in range(data.dl)] + [
                    x[:, :, cj, lj].sum() - 1 for lj in range(data.dl)] + [u[cj, :].sum()]
            elif cj == 1:
                cons = cons + [u[cj, :].sum() + np.einsum('kl,dkl->', data.csign, x[:, :, cj, :]) - 1] + [
                    u[cj, :].sum() + x[:, :, cj, :].sum() - 1]
            else:
                cons = cons + [u[cj, :].sum() + x[:, cj - 2, cj, :].sum() - 1]
        return cons

    @staticmethod
    def constraint_ineq(x, u, data):
        cons = list()
        for cj in range(data.dc):
            if cj >= 1:
                cons = cons + [-x[dj, kj, cj, lj] + data.v[dj, kj, cj, lj] * u[cj, lj] for dj in range(data.dd) for kj
                               in range(data.dk) for lj in range(data.dl)]
        return cons

    @staticmethod
    def solve(data):
        x0 = np.ones(data.dd * data.dk * data.dl * data.dc + data.dc * data.dl)
        cons = ({'type': 'ineq', 'fun': lambda x: SingleProblem.constraint_ineq(
            x[:data.dd * data.dk * data.dc * data.dl].reshape([data.dd, data.dk, data.dc, data.dl]),
            x[data.dd * data.dk * data.dc * data.dl:].reshape([data.dc, data.dl]), data)},
                {'type': 'eq', 'fun': lambda x: SingleProblem.constraint_eq(
                    x[:data.dd * data.dk * data.dc * data.dl].reshape([data.dd, data.dk, data.dc, data.dl]),
                    x[data.dd * data.dk * data.dc * data.dl:].reshape([data.dc, data.dl]), data)})

        bounds = ((0, 1),) * (data.dd * data.dk * data.dc * data.dl + data.dc * data.dl)
        fun = lambda x: SingleProblem.SP_obj(
            x[:data.dd * data.dk * data.dc * data.dl].reshape([data.dd, data.dk, data.dc, data.dl]),
            x[data.dd * data.dk * data.dc * data.dl:].reshape([data.dc, data.dl]), data)

        jac = lambda x: SingleProblem.SP_jacobian(
            x[:data.dd * data.dk * data.dc * data.dl].reshape([data.dd, data.dk, data.dc, data.dl]),
            x[data.dd * data.dk * data.dc * data.dl:].reshape([data.dc, data.dl]), data)
        hes = lambda x: SingleProblem.SP_hessian(
            x[:data.dd * data.dk * data.dc * data.dl].reshape([data.dd, data.dk, data.dc, data.dl]),
            x[data.dd * data.dk * data.dc * data.dl:].reshape([data.dc, data.dl]), data)
        res = opt.minimize(fun, np.ones(data.dd * data.dk * data.dl * data.dc + data.dc * data.dl), jac=jac,
                           bounds=bounds, constraints=cons, tol=0.0001, options={'disp': False})
        return (res.x[:data.dd * data.dk * data.dc * data.dl].reshape([data.dd, data.dk, data.dc, data.dl]))


def div0(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.true_divide(a, b)
        c[~np.isfinite(c)] = 0  # -inf, inf Nan
    return c


def fastroll(a):
    m, n = a.shape
    idx = np.mod((n - 1) * np.arange(m)[:, None] + np.arange(n), n)
    return a[np.arange(m)[:, None], idx]


def generateDailyProb(data, dynamic=None, cumulative=None, schedule=None, i=None, loc_dep=1, keepfiles=True):
    optim = JointOptimizer(keepfiles=keepfiles) if (data.joint is 1 or dynamic == "DCPP") else SingleOptimizer(
        keepfiles=keepfiles)

    y = None
    if dynamic in ["DP", "DCPP"]:
        if i>0:
            y = np.array([[[[tempGenerateY(i, data, schedule[k, c, loc, tj, max(0, i - data.dd):i, i:i + data.dd]) for
                             tj in range(data.dt)] for loc in
                            range(data.dl)] for c in range(data.dc)] for k in range(data.dk)])
    elif dynamic in ["CP"]:
        return randAllocation(None, data, dynamic, cum=cumulative, loc_dep=loc_dep)

    if y is None: y = np.zeros([data.dk, data.dc, data.dl, data.dt, data.dd-1, data.dd-1])
    x_out, c_out = optim.solve(data, dynamic, y, loc_dep)

    optim.del_tmpdir()

    x = np.zeros([data.dd, data.dk, data.dt, data.dc, data.dl])
    for rec in x_out:
        x[int(rec.key(0)) - 1, int(rec.key(1)) - 1, int(rec.key(2)) - 1, int(rec.key(3)) - 1, int(
            rec.key(4)) - 1] = rec.level
    if c_out is not 0:
        for rec in c_out:
            data.C[int(rec.key(0)) - 1, int(rec.key(1)) - 1] = rec.level
    # optim = SingleProblem()
    # x = optim.solve(data)
    # print(data.C)

    return randAllocation(x, data, dynamic, cumulative, loc_dep=loc_dep)


def tempGenerateY(t, data, schedule):
    ytemp = np.flip(schedule,0)
    temp = np.zeros([data.dd, data.dd])
    temp[:min(ytemp.shape[0], data.dd), :] = ytemp
    return temp


def randAllocation(P, data, dynamic=None, cum=None, loc_dep=1):
    if P is None:
        P = np.zeros([data.dd, data.dk, data.dt, data.dc, data.dl])
        for c in range(data.dc):
            if c is 0:
                P[:, :, :, c, :] = np.moveaxis(np.moveaxis(np.array(
                    [[data.v[:, :, tj, c, lc] / data.v[:, :, tj, c, lc].sum() for tj in range(data.dt)] for lc in
                     range(data.dl)]), 0, 3), 0, 2)
            elif c is 1:
                P[:, :, :, c, :] = data.v[:, :, :, c, :] / (
                        1 + data.v[:, :, :, c, :].sum()) if loc_dep is 1 else np.moveaxis(np.array(
                    [data.v[:, :, :, c, lc] / (1 + data.v[:, :, :, c, lc].sum()) for lc in range(data.dl)]), 0, 3)
            else:
                P[:, :, :, c, :] = np.moveaxis(
                    np.array([data.v[:, :, :, c, lc] / (1 + data.v[:, :, :, c, lc].sum()) for lc in range(data.dl)]), 0,
                    3)

    if cum is None:
        return P
    else:
        for c in range(data.dc):
            if c is 0:
                for lc in range(data.dl):
                    P[:, :, :, c, lc] = np.moveaxis(np.array(
                        [np.cumsum(np.ravel(P[:, :, tj, c, lc])).reshape([data.dd, data.dk]) for tj in range(data.dt)]),
                                                    0, 2)
            elif c is 1:
                P[:, :, :, c, :] = np.moveaxis(np.cumsum(np.ravel(np.moveaxis(P[:, :, :, c, :], 1, 2))).reshape(
                    [data.dd, data.dt, data.dk, data.dl]), 1, 2) if loc_dep is 1 \
                    else np.moveaxis(np.moveaxis(np.array(
                    [np.cumsum(np.ravel(np.moveaxis(P[:, :, :, c, lc], 1, 2))).reshape([data.dd, data.dt, data.dk]) for
                     lc in range(data.dl)]), 0, 3), 1, 2)
            else:
                for lc in range(data.dl):
                    P[:, :, :, c, lc] = np.moveaxis(
                        np.array([np.cumsum(P[:, k, :, c, lc]) for k in range(data.dk)]).reshape(
                            [data.dk, data.dd, data.dt]), 0, 1)
    return P.round(decimals=3)

