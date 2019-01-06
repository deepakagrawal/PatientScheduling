
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
Cy(kj,lj)       round_cap
y1(kj,lj)
z1(kj,lj)
z2(kj,lj)
alphaD(dj,kj,lj,nj)
P1D(dj,kj,lj,nj)
P2D(dj,kj,lj,nj)
P3D(dj,kj,lj,nj)
P4D(dj,kj,lj,nj)
;


Positive Variable x,u,C, Cy,alphaD,P1D, P2D, P3D,P4D;
Binary Variable cb;
x.up(dj,kj,tj,cj,lj) = sign(v(dj,kj,tj,cj,lj));;
u.up(cj,lj) = 1;
u.lo(cj,lj) = 0;
u.up(cj,lj)$(ord(cj) eq 1) = 0;
u.lo(cj,lj)$(ord(cj) >= 3 and ord(cj) <=5 and ord(lj) eq 2) = 1;
u.lo(cj,lj)$(ord(cj) >=6 and ord(lj) eq 1) = 1;
cb.up(kj,lj,i) = csign(kj,lj);
cb.up(kj,lj,i)$(ord(kj) <= 3 and ord(lj)=2) = 0;
cb.up(kj,lj,i)$(ord(kj)>=4 and ord(lj)=1) = 0;

Cy.up(kj,lj) = 1;

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
alphaEqD(dj,kj,lj,nj)
P1eqD(dj,kj,lj,nj)
P2eqD(dj,kj,lj,nj)
P3eqD(dj,kj,lj,nj)
P4eqD(dj,kj,lj,nj)
;




total_profit_DP         .. obj =e= sum((dj,kj,lj,nj), alphaD(dj,kj,lj,nj)*prob(nj))- 
                                   theta*sum((dj,kj,lj,nj),prob(nj)*(alphaD(dj,kj,lj,nj))*P1D(dj,kj,lj,nj) - C(kj,lj)*P2D(dj,kj,lj,nj) + C(kj,lj)*P3D(dj,kj,lj,nj) - alphaD(dj,kj,lj,nj)*P4D(dj,kj,lj,nj));

eq1(kj,cj,lj)$(ord(cj) >= 3 and ord(kj) eq ord(cj)-2)      .. u(cj,lj) + sum((dj,tj), x(dj,kj,tj,cj,lj)) =e= 1;
eq2_loc_dep_1(cj)$(ord(cj) eq 2)      .. sum(lj, u(cj,lj) + sum((kj,dj,tj),x(dj,kj,tj,cj,lj))) =e= 1;
eq3(dj,tj,cj,lj)$(ord(cj) eq 1 and ord(dj) eq 1)   .. sum(kj, x(dj,kj,tj,cj,lj)) =e= 1;
eq3_1(tj,cj,lj)$(ord(cj) eq 1) .. sum((kj,dj)$(ord(dj) >1), x(dj,kj,tj,cj,lj)) + u(cj,lj) =e= 0;
eq4(dj,kj,tj,cj,lj)$(ord(cj) eq 2)  .. x(dj,kj,tj,cj,lj)-v(dj,kj,tj,cj,lj)*sum(li$(ord(lj) eq ord(li)),u(cj,lj)) =l= 0;
eq4_1(dj,kj,tj,cj,lj)$(ord(cj) > 2 and ord(kj) eq ord(cj) -2)  .. x(dj,kj,tj,cj,lj)-v(dj,kj,tj,cj,lj)*u(cj,lj) =l= 0;
capCon1(kj,lj,i)$(ord(i)<=CAP)  .. cb(kj,lj,i) =g= cb(kj,lj,i+1);
capCon2(kj,lj)                 .. C(kj,lj) =e= sum(i$(ord(i)<=CAP+1),cb(kj,lj,i));
*capCon3(kj,lj,i)$(ord(i) eq 1) .. cb(kj,lj,i) =g= 0.001*sum((dj,tj,cj),x(dj,kj,tj,cj,lj));
*********** Dynamic
alphaEqD(dj,kj,lj,nj)   .. alphaD(dj,kj,lj,nj) =e= sum((cj,di,dx)$((ord(dx) eq ord(di)+ord(dj)-1) and (ord(di) ge 1) and (ord(di) le dd +1 - ord(dj))),sum(tj,y(kj,cj,lj,tj,di,dx))*rt(dx,di)) + 
sum(cj,L(cj,lj,nj)*r(dj)*sum(tj,x(dj,kj,tj,cj,lj))) + sum((cj,di)$((ord(di) ge 1) and (ord(di) le ord(dj)-1)),L(cj,lj,nj)*r(di)*sum(tj,z(di,kj,tj,cj,lj)));
P1eqD(dj,kj,lj,nj)      .. P1D(dj,kj,lj,nj) =e= 1 - exp(-alphaD(dj,kj,lj,nj))*(sum(i$(ord(i)<=CAP), cb(kj,lj,i+1)*alphaD(dj,kj,lj,nj)**(ord(i)-1)/gamma(ord(i)) ));
P2eqD(dj,kj,lj,nj)      .. P2D(dj,kj,lj,nj) =e= 1 - exp(-alphaD(dj,kj,lj,nj))*(sum(i$(ord(i)<=CAP+1), cb(kj,lj,i)*alphaD(dj,kj,lj,nj)**(ord(i)-1)/gamma(ord(i)) ));
P3eqD(dj,kj,lj,nj)      .. P3D(dj,kj,lj,nj) =e= exp(-alphaD(dj,kj,lj,nj))*(sum(i$(ord(i)<=CAP), cb(kj,lj,i+1)*alphaD(dj,kj,lj,nj)**(ord(i)-1)/gamma(ord(i)) ));
P4eqD(dj,kj,lj,nj)      .. P4D(dj,kj,lj,nj) =e= exp(-alphaD(dj,kj,lj,nj))*(sum(i$(ord(i)<=CAP-1), cb(kj,lj,i+2)*alphaD(dj,kj,lj,nj)**(ord(i)-1)/gamma(ord(i)) ));

MODEL DP_loc_dep /eq1,eq2_loc_dep_1,eq3,eq3_1,eq4,eq4_1,capCon1,capCon2,alphaEqD,P1eqD,P2eqD,P3eqD,P4eqD,total_profit_DP/;

option 
NLP = CONOPT
MINLP = SBB
limrow = 1
limcol = 1
*option solprint = off;
*option nodlim = 5000;
*option modeltype=convert;
optcr = .00
reslim = 5600;
DP_loc_dep.nodlim = 1000000;
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

solve DP_loc_dep using minlp maximizing obj;
*display gbdlog;    
    
