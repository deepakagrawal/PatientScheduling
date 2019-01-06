
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

*$if not set gdxincname $abort 'no include file name for data file provided'
$gdxin myDCP_d2.gdx
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
;


Positive Variable x,u,C;
Binary Variable cb;
x.up(dj,kj,tj,cj,lj) = sign(v(dj,kj,tj,cj,lj));;
u.up(cj,lj) = 1;
u.lo(cj,lj) = 0;
u.up(cj,lj)$(ord(cj) eq 1) = 0;
u.lo(cj,lj)$(ord(cj) >= 3 and ord(cj) <=5 and ord(lj) eq 2) = 1;
u.lo(cj,lj)$(ord(cj) >=6 and ord(lj) eq 1) = 1;
cb.up(kj,lj,i) = csign(kj,lj);

*cb.up(kj,lj,i) = sign(sum((dj,tj,cj),x(dj,kj,tj,cj,lj)));



*display u.up;

Equations
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
capCon3(kj,lj,i)
;




total_profit_DP         .. obj =e= -( sum((dj,kj,lj,nj), (sum((cj,di,dx)$((ord(dx) eq ord(di)+ord(dj)-1) and (ord(di) ge 1) and (ord(di) le dd +1 - ord(dj))),sum(tj,y(kj,cj,lj,tj,di,dx))*rt(dx,di)) + 
sum(cj,L(cj,lj,nj)*r(dj)*sum(tj,x(dj,kj,tj,cj,lj))) + sum((cj,di)$((ord(di) ge 1) and (ord(di) le ord(dj)-1)),L(cj,lj,nj)*r(di)*sum(tj,z(di,kj,tj,cj,lj))))*prob(nj))- 
                                   theta*sum((dj,kj,lj,nj),prob(nj)*((sum((cj,di,dx)$((ord(dx) eq ord(di)+ord(dj)-1) and (ord(di) ge 1) and (ord(di) le dd +1 - ord(dj))),sum(tj,y(kj,cj,lj,tj,di,dx))*rt(dx,di)) + 
sum(cj,L(cj,lj,nj)*r(dj)*sum(tj,x(dj,kj,tj,cj,lj))) + sum((cj,di)$((ord(di) ge 1) and (ord(di) le ord(dj)-1)),L(cj,lj,nj)*r(di)*sum(tj,z(di,kj,tj,cj,lj)))))*(1 - exp(-(sum((cj,di,dx)$((ord(dx) eq ord(di)+ord(dj)-1) and (ord(di) ge 1) and (ord(di) le dd +1 - ord(dj))),sum(tj,y(kj,cj,lj,tj,di,dx))*rt(dx,di)) + 
sum(cj,L(cj,lj,nj)*r(dj)*sum(tj,x(dj,kj,tj,cj,lj))) + sum((cj,di)$((ord(di) ge 1) and (ord(di) le ord(dj)-1)),L(cj,lj,nj)*r(di)*sum(tj,z(di,kj,tj,cj,lj)))))*(sum(i$(ord(i)<=CAP), cb(kj,lj,i+1)*(sum((cj,di,dx)$((ord(dx) eq ord(di)+ord(dj)-1) and (ord(di) ge 1) and (ord(di) le dd +1 - ord(dj))),sum(tj,y(kj,cj,lj,tj,di,dx))*rt(dx,di)) + 
sum(cj,L(cj,lj,nj)*r(dj)*sum(tj,x(dj,kj,tj,cj,lj))) + sum((cj,di)$((ord(di) ge 1) and (ord(di) le ord(dj)-1)),L(cj,lj,nj)*r(di)*sum(tj,z(di,kj,tj,cj,lj))))**(ord(i)-1)/gamma(ord(i)) ))) 
				   - C(kj,lj)*(1 - exp(-(sum((cj,di,dx)$((ord(dx) eq ord(di)+ord(dj)-1) and (ord(di) ge 1) and (ord(di) le dd +1 - ord(dj))),sum(tj,y(kj,cj,lj,tj,di,dx))*rt(dx,di)) + 
sum(cj,L(cj,lj,nj)*r(dj)*sum(tj,x(dj,kj,tj,cj,lj))) + sum((cj,di)$((ord(di) ge 1) and (ord(di) le ord(dj)-1)),L(cj,lj,nj)*r(di)*sum(tj,z(di,kj,tj,cj,lj)))))*(sum(i$(ord(i)<=CAP+1), cb(kj,lj,i)*(sum((cj,di,dx)$((ord(dx) eq ord(di)+ord(dj)-1) and (ord(di) ge 1) and (ord(di) le dd +1 - ord(dj))),sum(tj,y(kj,cj,lj,tj,di,dx))*rt(dx,di)) + 
sum(cj,L(cj,lj,nj)*r(dj)*sum(tj,x(dj,kj,tj,cj,lj))) + sum((cj,di)$((ord(di) ge 1) and (ord(di) le ord(dj)-1)),L(cj,lj,nj)*r(di)*sum(tj,z(di,kj,tj,cj,lj))))**(ord(i)-1)/gamma(ord(i)) ))) 
				   + C(kj,lj)*(exp(-(sum((cj,di,dx)$((ord(dx) eq ord(di)+ord(dj)-1) and (ord(di) ge 1) and (ord(di) le dd +1 - ord(dj))),sum(tj,y(kj,cj,lj,tj,di,dx))*rt(dx,di)) + 
sum(cj,L(cj,lj,nj)*r(dj)*sum(tj,x(dj,kj,tj,cj,lj))) + sum((cj,di)$((ord(di) ge 1) and (ord(di) le ord(dj)-1)),L(cj,lj,nj)*r(di)*sum(tj,z(di,kj,tj,cj,lj)))))*(sum(i$(ord(i)<=CAP), cb(kj,lj,i+1)*(sum((cj,di,dx)$((ord(dx) eq ord(di)+ord(dj)-1) and (ord(di) ge 1) and (ord(di) le dd +1 - ord(dj))),sum(tj,y(kj,cj,lj,tj,di,dx))*rt(dx,di)) + 
sum(cj,L(cj,lj,nj)*r(dj)*sum(tj,x(dj,kj,tj,cj,lj))) + sum((cj,di)$((ord(di) ge 1) and (ord(di) le ord(dj)-1)),L(cj,lj,nj)*r(di)*sum(tj,z(di,kj,tj,cj,lj))))**(ord(i)-1)/gamma(ord(i)) ))) 
				   - (sum((cj,di,dx)$((ord(dx) eq ord(di)+ord(dj)-1) and (ord(di) ge 1) and (ord(di) le dd +1 - ord(dj))),sum(tj,y(kj,cj,lj,tj,di,dx))*rt(dx,di)) + 
sum(cj,L(cj,lj,nj)*r(dj)*sum(tj,x(dj,kj,tj,cj,lj))) + sum((cj,di)$((ord(di) ge 1) and (ord(di) le ord(dj)-1)),L(cj,lj,nj)*r(di)*sum(tj,z(di,kj,tj,cj,lj))))*(exp(-(sum((cj,di,dx)$((ord(dx) eq ord(di)+ord(dj)-1) and (ord(di) ge 1) and (ord(di) le dd +1 - ord(dj))),sum(tj,y(kj,cj,lj,tj,di,dx))*rt(dx,di)) + 
sum(cj,L(cj,lj,nj)*r(dj)*sum(tj,x(dj,kj,tj,cj,lj))) + sum((cj,di)$((ord(di) ge 1) and (ord(di) le ord(dj)-1)),L(cj,lj,nj)*r(di)*sum(tj,z(di,kj,tj,cj,lj)))))*(sum(i$(ord(i)<=CAP-1), cb(kj,lj,i+2)*(sum((cj,di,dx)$((ord(dx) eq ord(di)+ord(dj)-1) and (ord(di) ge 1) and (ord(di) le dd +1 - ord(dj))),sum(tj,y(kj,cj,lj,tj,di,dx))*rt(dx,di)) + 
sum(cj,L(cj,lj,nj)*r(dj)*sum(tj,x(dj,kj,tj,cj,lj))) + sum((cj,di)$((ord(di) ge 1) and (ord(di) le ord(dj)-1)),L(cj,lj,nj)*r(di)*sum(tj,z(di,kj,tj,cj,lj))))**(ord(i)-1)/gamma(ord(i)) )))));
eq1(kj,cj,lj)$(ord(cj) >= 3 and ord(kj) eq ord(cj)-2)      .. u(cj,lj) + sum((dj,tj), x(dj,kj,tj,cj,lj)) =e= 1;
eq2_loc_dep_1(cj)$(ord(cj) eq 2)      .. sum(lj, u(cj,lj) + sum((kj,dj,tj),x(dj,kj,tj,cj,lj))) =e= 1;
eq3(dj,tj,cj,lj)$(ord(cj) eq 1 and ord(dj) eq 1)   .. sum(kj, x(dj,kj,tj,cj,lj)) =e= 1;
eq3_1(tj,cj,lj)$(ord(cj) eq 1) .. sum((kj,dj)$(ord(dj) >1), x(dj,kj,tj,cj,lj)) + u(cj,lj) =e= 0;
eq4(dj,kj,tj,cj,lj)$(ord(cj) eq 2)  .. x(dj,kj,tj,cj,lj)-v(dj,kj,tj,cj,lj)*sum(li$(ord(lj) eq ord(li)),u(cj,lj)) =l= 0;
eq4_1(dj,kj,tj,cj,lj)$(ord(cj) > 2 and ord(kj) eq ord(cj) -2)  .. x(dj,kj,tj,cj,lj)-v(dj,kj,tj,cj,lj)*u(cj,lj) =l= 0;
capCon1(kj,lj,i)$(ord(i)<=CAP+1)  .. cb(kj,lj,i) =g= cb(kj,lj,i+1);
capCon2(kj,lj)                 .. C(kj,lj) =e= sum(i$(ord(i)<=CAP+1),cb(kj,lj,i));
capCon3(kj,lj,i) .. cb(kj,lj,i) =l= csign(kj,lj);
 


MODEL DP_loc_dep /eq1,eq2_loc_dep_1,eq3,eq3_1,eq4,eq4_1,capCon1,capCon2,capCon3,total_profit_DP/;

option MINLP = SBB;
*options NLP = minos;
option limrow = 0;
option limcol = 0;
option solprint = off;
option threads = 8;
*option nodlim = 5000;
*option modeltype=convert;
option optcr = 0;
DP_loc_dep.nodlim = 10000;
DP_loc_dep.optfile = 1;
$onecho >SBB.opt
*epint 0.1
*mipinterval 1
*bfsstay 5
*userheurfirst 100
*nodesel 0
loginterval 10
*varsel 1
$offecho

*solve DP_loc_dep using rminlp maximizing obj;

*-----------------------------------
* GBD sub problem
* this is the NLP model resulting from fixing the integer variables.

model sub /eq1,eq2_loc_dep_1, eq3, eq3_1, eq4, eq4_1, capCon2, total_profit_DP/;

*-----------------------------------
* GBD master problem

set iter 'cycle numbers' /iter1*iter10/;
set cycle(iter) 'dynamic subset: cycles done';

variable mu;
equation mastercut(iter);
equation integercut(iter);
parameter lambda(iter,kj,lj) 'duelas of equation capCon3';
parameter xkeep(iter,dj,kj,tj,cj,lj) 'continous variables from previous cycles';
parameter ukeep(iter, cj,lj) 'continous variables from previous cycles';
set icutzeroCb(iter, kj,lj,i);
*set icutzeroU(iter, cj,lj);
set icutoneCb(iter, kj,lj,i);
set icutset(iter);
*set icutoneX(iter, cj,lj);

mastercut(cycle)..
	mu =g= -( sum((dj,kj,lj,nj), (sum((cj,di,dx)$((ord(dx) eq ord(di)+ord(dj)-1) and (ord(di) ge 1) and (ord(di) le dd +1 - ord(dj))),sum(tj,y(kj,cj,lj,tj,di,dx))*rt(dx,di)) + 
sum(cj,L(cj,lj,nj)*r(dj)*sum(tj,xkeep(cycle,dj,kj,tj,cj,lj))) + sum((cj,di)$((ord(di) ge 1) and (ord(di) le ord(dj)-1)),L(cj,lj,nj)*r(di)*sum(tj,z(di,kj,tj,cj,lj))))*prob(nj))- 
                                   theta*sum((dj,kj,lj,nj),prob(nj)*((sum((cj,di,dx)$((ord(dx) eq ord(di)+ord(dj)-1) and (ord(di) ge 1) and (ord(di) le dd +1 - ord(dj))),sum(tj,y(kj,cj,lj,tj,di,dx))*rt(dx,di)) + 
sum(cj,L(cj,lj,nj)*r(dj)*sum(tj,xkeep(cycle,dj,kj,tj,cj,lj))) + sum((cj,di)$((ord(di) ge 1) and (ord(di) le ord(dj)-1)),L(cj,lj,nj)*r(di)*sum(tj,z(di,kj,tj,cj,lj)))))*(1 - exp(-(sum((cj,di,dx)$((ord(dx) eq ord(di)+ord(dj)-1) and (ord(di) ge 1) and (ord(di) le dd +1 - ord(dj))),sum(tj,y(kj,cj,lj,tj,di,dx))*rt(dx,di)) + 
sum(cj,L(cj,lj,nj)*r(dj)*sum(tj,xkeep(cycle,dj,kj,tj,cj,lj))) + sum((cj,di)$((ord(di) ge 1) and (ord(di) le ord(dj)-1)),L(cj,lj,nj)*r(di)*sum(tj,z(di,kj,tj,cj,lj)))))*(sum(i$(ord(i)<=CAP), cb(kj,lj,i+1)*(sum((cj,di,dx)$((ord(dx) eq ord(di)+ord(dj)-1) and (ord(di) ge 1) and (ord(di) le dd +1 - ord(dj))),sum(tj,y(kj,cj,lj,tj,di,dx))*rt(dx,di)) + 
sum(cj,L(cj,lj,nj)*r(dj)*sum(tj,xkeep(cycle,dj,kj,tj,cj,lj))) + sum((cj,di)$((ord(di) ge 1) and (ord(di) le ord(dj)-1)),L(cj,lj,nj)*r(di)*sum(tj,z(di,kj,tj,cj,lj))))**(ord(i)-1)/gamma(ord(i)) ))) 
				   - C(kj,lj)*(1 - exp(-(sum((cj,di,dx)$((ord(dx) eq ord(di)+ord(dj)-1) and (ord(di) ge 1) and (ord(di) le dd +1 - ord(dj))),sum(tj,y(kj,cj,lj,tj,di,dx))*rt(dx,di)) + 
sum(cj,L(cj,lj,nj)*r(dj)*sum(tj,xkeep(cycle,dj,kj,tj,cj,lj))) + sum((cj,di)$((ord(di) ge 1) and (ord(di) le ord(dj)-1)),L(cj,lj,nj)*r(di)*sum(tj,z(di,kj,tj,cj,lj)))))*(sum(i$(ord(i)<=CAP+1), cb(kj,lj,i)*(sum((cj,di,dx)$((ord(dx) eq ord(di)+ord(dj)-1) and (ord(di) ge 1) and (ord(di) le dd +1 - ord(dj))),sum(tj,y(kj,cj,lj,tj,di,dx))*rt(dx,di)) + 
sum(cj,L(cj,lj,nj)*r(dj)*sum(tj,xkeep(cycle,dj,kj,tj,cj,lj))) + sum((cj,di)$((ord(di) ge 1) and (ord(di) le ord(dj)-1)),L(cj,lj,nj)*r(di)*sum(tj,z(di,kj,tj,cj,lj))))**(ord(i)-1)/gamma(ord(i)) ))) 
				   + C(kj,lj)*(exp(-(sum((cj,di,dx)$((ord(dx) eq ord(di)+ord(dj)-1) and (ord(di) ge 1) and (ord(di) le dd +1 - ord(dj))),sum(tj,y(kj,cj,lj,tj,di,dx))*rt(dx,di)) + 
sum(cj,L(cj,lj,nj)*r(dj)*sum(tj,xkeep(cycle,dj,kj,tj,cj,lj))) + sum((cj,di)$((ord(di) ge 1) and (ord(di) le ord(dj)-1)),L(cj,lj,nj)*r(di)*sum(tj,z(di,kj,tj,cj,lj)))))*(sum(i$(ord(i)<=CAP), cb(kj,lj,i+1)*(sum((cj,di,dx)$((ord(dx) eq ord(di)+ord(dj)-1) and (ord(di) ge 1) and (ord(di) le dd +1 - ord(dj))),sum(tj,y(kj,cj,lj,tj,di,dx))*rt(dx,di)) + 
sum(cj,L(cj,lj,nj)*r(dj)*sum(tj,xkeep(cycle,dj,kj,tj,cj,lj))) + sum((cj,di)$((ord(di) ge 1) and (ord(di) le ord(dj)-1)),L(cj,lj,nj)*r(di)*sum(tj,z(di,kj,tj,cj,lj))))**(ord(i)-1)/gamma(ord(i)) ))) 
				   - (sum((cj,di,dx)$((ord(dx) eq ord(di)+ord(dj)-1) and (ord(di) ge 1) and (ord(di) le dd +1 - ord(dj))),sum(tj,y(kj,cj,lj,tj,di,dx))*rt(dx,di)) + 
sum(cj,L(cj,lj,nj)*r(dj)*sum(tj,xkeep(cycle,dj,kj,tj,cj,lj))) + sum((cj,di)$((ord(di) ge 1) and (ord(di) le ord(dj)-1)),L(cj,lj,nj)*r(di)*sum(tj,z(di,kj,tj,cj,lj))))*(exp(-(sum((cj,di,dx)$((ord(dx) eq ord(di)+ord(dj)-1) and (ord(di) ge 1) and (ord(di) le dd +1 - ord(dj))),sum(tj,y(kj,cj,lj,tj,di,dx))*rt(dx,di)) + 
sum(cj,L(cj,lj,nj)*r(dj)*sum(tj,xkeep(cycle,dj,kj,tj,cj,lj))) + sum((cj,di)$((ord(di) ge 1) and (ord(di) le ord(dj)-1)),L(cj,lj,nj)*r(di)*sum(tj,z(di,kj,tj,cj,lj)))))*(sum(i$(ord(i)<=CAP-1), cb(kj,lj,i+2)*(sum((cj,di,dx)$((ord(dx) eq ord(di)+ord(dj)-1) and (ord(di) ge 1) and (ord(di) le dd +1 - ord(dj))),sum(tj,y(kj,cj,lj,tj,di,dx))*rt(dx,di)) + 
sum(cj,L(cj,lj,nj)*r(dj)*sum(tj,xkeep(cycle,dj,kj,tj,cj,lj))) + sum((cj,di)$((ord(di) ge 1) and (ord(di) le ord(dj)-1)),L(cj,lj,nj)*r(di)*sum(tj,z(di,kj,tj,cj,lj))))**(ord(i)-1)/gamma(ord(i)) )))))+
	sum((kj,lj), lambda(cycle, kj,lj)*C(kj,lj) - sum(i$(ord(i)<=CAP+1),cb(kj,lj,i)));

integercut(icutset)..
	sum(icutoneCb(icutset,kj,lj,i), cb(kj,lj,i)) - 
	sum(icutzeroCb(icutset, kj,lj,i), cb(kj,lj,i)) =l= sum(icutoneCb(icutset, kj,lj,i),1)-1;

icutset(iter) = no;
icutzeroCb(iter,kj,lj,i) = no;
icutoneCb(iter, kj,lj,i) = no;

model master /mastercut, integercut, capCon1,capCon3/;
*-----------------------------
* initial integer configuration

cb.l(kj,lj,i) = 0;
cb.l(kj,lj,i)$(ord(i) <= 5) = csign(kj,lj);

*------------------------------
* Generalized Benders Decomposition

option minlp = SBB;
option rminlp = conopt;

set problem /sub,master/;
parameter gdblog(iter, problem, *) 'algorithm log';
parameter cb_best(kj,lj,i) 'record best cap';
parameter xbest(dj,kj,tj,cj,lj) 'record best solution';
parameter C_best(kj,lj) 'record best capacity';
parameter u_best(cj,lj) 'record best u';

*
* initialization
*
scalar UB 'upperbound' /INF/;
scalar done /0/;

loop(iter$(not done),
*
* fix integer variables
*
  cb.fx(kj,lj,i) = round(cb.l(kj,lj,i));
  display cb.l;

*
* solve NLP subproblem
*
  solve sub minimizing obj using rminlp;
  abort$(sub.solvestat <>1) "NLP Solver did not return OK";
  abort$(sub.modelstat = 3 or sub.modelstat >= 6) "NLP Solver did not return OK";

  gdblog(iter, 'sub', 'solvestat') = sub.solvestat;
  gdblog(iter, 'sub', 'modelstat') = sub.modelstat;

*
* if optimal/locally optimal
*	

  if(sub.modelstat = 1 or sub.modelstat = 2,
  		gdblog(iter, 'sub', 'obj') = obj.l;
*
* do we have a better solution?
*
	  if(obj.l < UB,
	  	 UB = obj.l;
	  	 gdblog(iter, 'sub', 'better') = 1;
*
* record best solutioon
*
		 cb_best(kj,lj,i) = cb.l(kj,lj,i);
		 xbest(dj,kj,tj,cj,lj) = x.l(dj,kj,tj,cj,lj);
		 C_best(kj,lj) = C.l(kj,lj);
		 u_best(cj,lj) = u.l(cj,lj);
	  	 );
	);
*
* remember data for constructin Bender's cuts
*

	xkeep(iter, dj,kj,tj,cj,lj) = x.l(dj,kj,tj,cj,lj);
	ukeep(iter, cj,lj) = u.l(cj,lj);
	lambda(iter, kj,lj) = - capCon2.m(kj,lj);
	cycle(iter) = yes; 
*
* remove fixed bounds
*
	cb.lo(kj,lj,i) = 0;
	cb.up(kj,lj,i) = 1;

*
* solve master problem
*
	solve master minimizing mu using minlp;
	gdblog(iter, 'master', 'solvestat') = master.solvestat;
	gdblog(iter, 'master', 'modelstat') = master.modelstat;
	gdblog(iter, 'master', 'obj') = mu.l;

*
* make sure this integer solution is not visited again
*
	icutset(iter) = yes;
	icutzeroCb(iter, kj,lj,i)$(cb.l(kj,lj,i)<0.5) = yes;
	icutoneCb(iter, kj,lj,i)$(cb.l(kj,lj,i)>0.5) = yes;

*
* stopping criterion
*
	if (mu.l>= UB,
		done = 1;
		);
);

display gdblog;
display xbest, u_best, C_best;