from __future__ import division
from pyutilib.misc import Options
import multiprocessing as mp
import sys,getopt
import sys
import pandas as pd
from model_code1 import generateDailyProb
import numpy as np
import itertools
import tempfile
from scipy import stats
import functools
from gams import *
import argparse
import json


def allocateUrgent(pat,P,patgoing,cancel,left,cr):
    ## Allocate Urgent Patients at different location independenlty
    temp = np.random.random([pat,2])
    for j in range(pat):
        done = 0
        if temp[j,0]>P.max():
            left += 1
        else:
            for k in range(data.dk):
                if temp[j,0] < P[k] and done is 0:
                    patgoing[k] = patgoing[k] + 1
                    if temp[j,1] < 1 - cr:
                        cancel[k] = cancel[k] + 1
                    done = 1
                    break
    return (patgoing,cancel,left)


def allocateFlex_dep(pat,P,patgoing,cancel,left,cr):
    ## Allocate Flexible Patients at different location dependenlty (i.e. sum over all location should be 1)
    temp = np.random.random([pat, 2])
    for j in range(pat):
        done =0
        if temp[j,0]>P.max():
            left += 1 ### Pooled left patients in location 0
        else:
            for idx in range(data.dd):
                if done == 1:
                    break
                for tj in range(data.dt):
                    if done == 1:
                        break
                    for k in range(data.dk):
                        if done == 1:
                            break
                        else:
                            for loc in range(data.dl):
                                if temp[j,0] < P[idx, k, tj,loc]:
                                    patgoing[idx, k,tj, loc] = patgoing[idx, k,tj, loc] + 1
                                    if temp[j,1] < 1 - cr[idx]:
                                        cancel[idx, k,tj,loc] = cancel[idx, k,tj,loc] + 1
                                    done = 1
                                    break
    return (patgoing,cancel,left)

def allocateFlex_ind(pat,P,patgoing,cancel,left,cr):
    ## Allocate Flexible Patients at different location dependenlty (i.e. sum over all location should be 1)
    temp = np.random.random([pat, 2])
    for j in range(pat):
        done =0
        if temp[j,0]>P.max():
            left += 1 ### Pooled left patients in location 0
        else:
            for idx in range(data.dd):
                if done == 1:
                    break
                for tj in range(data.dt):
                    if done ==1:
                        break
                    for k in range(data.dk):
                        if done == 1:
                            break
                        if temp[j,0] < P[idx, k, tj]:
                            patgoing[idx, k, tj] = patgoing[idx, k, tj] + 1
                            if temp[j,1] < 1 - cr[idx]:
                                cancel[idx, k, tj] = cancel[idx, k, tj] + 1
                            done = 1
                            break
    return (patgoing,cancel,left)

def allocateDedicated(pat,P,patgoing,cancel,left,cr):
    ## Allocate Dedicated Patients at different location independenlty
    temp = np.random.random([pat, 2])
    for j in range(pat):
        done = 0
        if temp[j,0]>P.max():
            left += 1
        else:
            for idx in range(data.dd):
                if done ==1:
                    break
                for tj in range(data.dt):
                    if temp[j,0] < P[idx, tj]:
                        patgoing[idx,tj] = patgoing[idx,tj] + 1
                        if temp[j,1] < 1 - cr[idx]:
                            cancel[idx,tj] = cancel[idx,tj] + 1
                        done = 1
                        break
    return (patgoing,cancel,left)

# def addFutureDemand(lam,choice):
#     dd,dk,dc,dl = choice.shape
#     y = np.zeros(dl)
#     x = np.zeros(choice.shape)
#     for cj in range(dc):
#         for lj in range(dl):
#             if cj==0:
#                 x[:,:,cj,lj] = lam[cj,lj]*choice[:,:,cj,lj]#*(choice[:,:,cj,lj]/choice[:,:,cj,lj].sum())
#             elif cj==1:
#                 x[:, :, cj, lj] = lam[cj, lj]*choice[:,:,cj,lj] #* (choice[:, :, cj, lj] / (1+choice[:, :, cj, lj].sum()))
#             else:
#                 x[:, cj-2, cj, lj] = lam[cj, lj]*choice[:,cj-2,cj,lj] #* (choice[:, cj-2, cj, lj] / (1 + choice[:, cj-2, cj, lj].sum()))
#
#     for i in range(1,dd):
#         y = y + np.apply_over_axes(np.sum,np.moveaxis(x.sum(axis=2).reshape([dd,dk,dl]),0,2)[:,:,:-i],axes=[0,2]).reshape(dl)
#
#     return y
#
#
# def DQP_factor(data,sum_schedule):
#     #sum_schedule = sum_schedule+addFutureDemand(data.l,data.v)
#     return (data.C.sum(axis=0) / sum_schedule) / (data.C.sum(axis=0) / sum_schedule).sum()


def run_main(iter, data=None):
    ## Run simulation for location dependent scenario
    np.random.seed(iter)
    schedule = np.zeros([data.dk, data.dc, data.dl,data.dt,data.Tmax, data.Tmax + data.dd - 1])
    patleft = np.zeros(data.Tmax)
    pat = np.array(
        [[np.random.poisson(data.ldist[c, loc].rvs(random_state=iter), size=data.Tmax) for loc in range(data.dl)] for c in
         range(data.dc)])
    pat[1,:] = pat[1,:].sum(axis=0) if loc_dep is 1 else pat[1,:]
    for i in range(data.Tmax):
        if data.dynamic in ["DP","DPP","DCPP"]:
            P = generateDailyProb(data, data.dynamic, 1, schedule, i, loc_dep=loc_dep)

        # elif data.dynamic in ["DQP"]: ###### Try with adding future expected appointments
        #     if i>0:
        #         # data.l[1,:] = flex_data*(schedule.sum()/np.apply_over_axes(np.sum, schedule, axes=[0,1,3,4]).reshape([data.dl,1]))
        #         data.l[1, :] = flex_data * DQP_factor(data, np.apply_over_axes(np.sum, schedule, axes=[0, 1, 3, 4]).reshape([data.dl])).reshape([data.dl,1])
        #     else:
        #         data.l[1, :] = data.l[1, 0] * (data.C.sum(axis=0) / data.C.sum()).reshape(data.dl,1)
        #     P = generateDailyProb(data, "DPP", 1, schedule, i, loc_dep=0)
        else:
            P = data.Prob
        data.dat_C = data.dat_C.append({'Run':iter, 'Horizon':data.dd, 'Theta':data.tehta, 'Choice':data.ch, 'Policy':data.dynamic,
                      'Loc_Dep':loc_dep, 'Joint':joint[0], 'Doubly':dob, 'day':i, 'C11':data.C[0,0], 'C12':data.C[0,1],
                      'C21':data.C[1,0], 'C22':data.C[1,1], 'C31':data.C[2,0], 'C32':data.C[2,1], 'C41':data.C[3,0],
                      'C42':data.C[3,1], 'C51':data.C[4,0], 'C52':data.C[4,1]}, ignore_index=True)
        patgoing = np.zeros([data.dd, data.dk, data.dt,data.dc,data.dl])
        cancel = np.zeros([data.dd, data.dk,data.dt, data.dc,data.dl])
        left = np.zeros([data.dc,data.dl])

        for c in range(data.dc):
            if c is 0:
                for loc in range(data.dl):
                    for tj in range(data.dt):
                        patgoing[0, :, tj,c, loc], cancel[0, :, tj,c, loc], left[c, loc] = allocateUrgent(pat[c, loc, i],
                                                                                                    P[0, :,tj, c, loc],
                                                                                                    patgoing[0, :, tj,c, loc],
                                                                                                    cancel[0, :, tj,c, loc],
                                                                                                    left[c, loc], data.r[0])
                        if patgoing[0,:,tj,c,loc].sum() + left[c,loc] != pat[c,loc,i]:
                            raise ValueError('incoming patients are not equal to scheduled + left')
            elif c is 1:
                if loc_dep is 1: # Location Dependent
                    patgoing[:,:,:,c,:], cancel[:,:,:,c,:],left[c,:]=allocateFlex_dep(pat[c,0,i], P[:,:,:,c,:], patgoing[:,:,:,c,:], cancel[:,:,:,c,:],left[c,:],data.r)
                    if patgoing[:, :,:, c, :].sum() + left[c, 0] != pat[c, 0, i]:
                        raise ValueError('incoming patients are not equal to scheduled + left')
                else: # Location independent
                    for loc in range(data.dl):
                        patgoing[:,:,:,c,loc], cancel[:,:,:,c,loc],left[c,loc]=allocateFlex_ind(pat[c,loc,i], P[:,:,:,c,loc], patgoing[:,:,:,c,loc], cancel[:,:,:,c,loc],left[c,loc],data.r)
                        if patgoing[:, :, :,c, loc].sum() + left[c,loc].sum() != pat[c, loc, i].sum():
                            raise ValueError('incoming patients are not equal to scheduled + left')
            else:
                for loc in range(data.dl):
                    patgoing[:, c - 2,:, c, loc], cancel[:, c - 2,:, c, loc], left[c, loc] = allocateDedicated(
                        pat[c, loc, i], P[:, c - 2,:, c, loc], patgoing[:, c - 2,:, c, loc], cancel[:, c - 2,:, c, loc],
                        left[c, loc], data.r)
                    if patgoing[:,c-2,:,c,loc].sum() + left[c,loc] != pat[c,loc,i]:
                        raise ValueError('incoming patients are not equal to scheduled + left')

        patfinal = patgoing - cancel
        schedule[:, :,:,:, i, i:i + data.dd] = np.moveaxis(np.moveaxis(patfinal, 0, 4),1,3)
        patleft[i] = left.sum()

    appSum = np.apply_over_axes(np.sum,schedule,axes=[4, 3,1]).squeeze()#schedule.sum(axis=2).sum(axis=1)
    profit = np.apply_over_axes(np.sum, appSum, axes=[0,1]).squeeze() - data.theta*np.array([np.abs(appSum[:,:,kk] - data.C).sum() for kk in range(appSum.shape[2])])

    return (profit[data.day0:data.Tmax].sum() / (data.Tmax - data.day0), patleft[data.day0:data.Tmax].sum()/(data.Tmax - data.day0), data.dat_C)

def calcFun(dk,dl,end,demand):
    arr = np.zeros([end,dk,dl])
    for j,k in itertools.product(range(dk),range(dl)):
        arr[:,j,k] = [max(1.5, demand[j,k] - 0.5*i) if demand[j,k]!=0 else 0 for i in range(end)]
    return arr

def calculate_choice(dd,dk,dt,dc,dl,ch,varray):
    v = np.zeros([dd, dk, dt, dc, dl])
    tmp_v = varray[ch][:dd, :, :]
    for c in range(dc):
        if c is 0:
            v[0, :, :,c, :] = np.moveaxis(np.array([tmp_v[0, :, :] for t in range(dt)]), 0, 1)
        elif c is 1:
            v[:, :, :, c, :] = np.moveaxis(np.array([tmp_v for t in range(dt)]),0,2)
        else:
            v[:, c - 2, :, c, :] = np.moveaxis(np.array([tmp_v[:, c - 2, :] for t in range(dt)]),0,1)
    return v

def calc_lambda(data,doubly):
    data.lprob = 1 if doubly is 0 else 3
    data.probl = [1.0] if doubly is 0 else (1.0/data.lprob)*np.ones(data.lprob)
    data.l = np.zeros([data.dc, data.dl,data.lprob])  # data.l=np.array([[2, 2, 6, 0], [2, 2, 0, 6]])
    for i in range(data.lprob):
        data.l[0,:,i] = np.multiply(temp_demand,(1-(temp_new+temp_returning))).sum(axis=0) + \
                        np.multiply(np.sign(np.multiply(temp_demand,(1-(temp_new+temp_returning))).sum(axis=0)),
                                    ((data.lprob-1.0)/2-i)*(np.multiply(temp_demand,(1-(temp_new+temp_returning))).sum(axis=0))*0.1)
        data.l[1,:,i] = np.multiply(temp_demand,temp_new).sum(axis=0) +\
                        np.multiply(np.sign(np.multiply(temp_demand,temp_new).sum(axis=0)),
                                    ((data.lprob - 1.0) / 2 - i) * (np.multiply(temp_demand, (1 - (temp_new + temp_returning))).sum(axis=0)) * 0.1)
        data.l[2:,:,i]= np.multiply(temp_demand,temp_returning) + \
                        np.multiply(np.sign(np.multiply(temp_demand,temp_returning).sum(axis=0)),
                                    ((data.lprob-1.0)/2-i)*(np.multiply(temp_demand,(1-(temp_new+temp_returning))).sum(axis=0))*0.1)
    data.simL = np.copy(data.l)
    data.simL[0,:,:] = data.simL[0,:,:]/data.dt
    data.ldist = np.array(
        [[stats.rv_discrete(values=(data.simL[i, j, :], data.probl)) for j in range(data.dl)] for i in range(data.dc)])
    l_original = np.copy(data.l[1, :, :])
    return l_original

if __name__ == "__main__":
    # if len(sys.argv) == 1:
    #     Imax, processes, horizon, theta, dchoice_a, dchoice_b, loc_a, loc_b, policy, joint, doubly, nslots\
    #         = 1, 1, [2], [1.25], 1, 2, 1, 2, ["DCPP"], [1], [0], 2
    # else:
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--Imax", help="Maximum Number of Iterations", type=int, default=1)
    parser.add_argument('-p', "--processes", help="Maximum Number of parallel processes to run", type=int, default=1)
    parser.add_argument('-s', "--horizon", help="List of maximum number of days", type=int, nargs='+', default=2)
    parser.add_argument('-t', "--theta", help="Overtime Undertime Penalty", type=float, nargs='+', default=1.25)
    parser.add_argument('-c', "--choices", help="Choices", type=int, nargs='+', default=1)
    parser.add_argument('-l', "--locations", help="Locations", type=int, nargs='+', default=1)
    parser.add_argument('-o', "--policies", help="Policies to run", nargs='+', default="CP")
    parser.add_argument('-j', '--joint', help="Run Joint Optimization Scenario", type=int, default=0)
    parser.add_argument('-d', "--doubly", help="Doubly Stochastic or not", type=int, default=0)
    parser.add_argument('-n', "--nslots", help="Number of slots to run", type=int, default=1)
    args = parser.parse_args()
        # myopts, args = getopt.getopt(sys.argv[1:], 'i:p:s:t:c:l:o:j:d:n:')
        # for o,a in myopts:
        #     if o == "-i":
        #         Imax = int(a)
        #     elif o=="-p":
        #         processes = int(a)
        #     elif o== "-s":
        #         horizon = [int(i) for i in a.split(",")]
        #     elif o=="-t":
        #         theta = [float(i) for i in a.split(",")]
        #     elif o=="-c":
        #         dchoice_a,dchoice_b = [int(i) for i in a.split(",")]
        #     elif o=="-l":
        #         loc_a,loc_b = [int(i) for i in a.split(",")]
        #     elif o=="-o":
        #         policy = a.split(",")
        #     elif o=="-j":
        #         joint = [int(a)]
        #     elif o== "-d":
        #         doubly = [int(a)]
        #     else:
        #         nslots = int(a)

    dat = pd.read_csv('new_dem_cap_loc_data.csv')

    data = Options()   # Create data structure
    data.Tmax = 30
    data.day0 = 15
    data.dk = dat.Provider.unique().shape[0]
    data.dt = args.nslots
    data.CAP = 30

    data.dc = data.dk + 2
    data.dl = dat.Clinic.unique().shape[0]
    temp_demand = dat[["Provider","Clinic","Daily_Demand"]].pivot(index='Provider', columns='Clinic',values='Daily_Demand').fillna(0).as_matrix()
    temp_new = dat[["Provider","Clinic","New"]].pivot(index='Provider', columns='Clinic',values='New').fillna(0).as_matrix()
    temp_returning = dat[["Provider", "Clinic", "Returning"]].pivot(index='Provider', columns='Clinic', values='Returning').fillna(0).as_matrix()




    data.C = dat[['Provider','Clinic','Capacity']].pivot(index='Provider',columns = 'Clinic',values = 'Capacity').fillna(0).as_matrix()
    # data.C = np.moveaxis(np.array([data.C1/data.dt for i in range(data.dt)]),0,-1)
    data.C = np.round(data.C,0)
    Cold = data.C.copy()
    data.csign = np.sign(data.C)
    data.beta = np.zeros(data.dc)
    data.beta[:2] = 0
    data.gamma = np.ones(data.dk)


    step = 5

    # horizon = range(start, end, step)
    demand = dat[["Provider", "Clinic", "Daily_Demand"]].pivot(index="Provider", columns="Clinic",
                                                                    values="Daily_Demand").fillna(0).as_matrix()
    tmp_demand = np.tile(demand, (args.horizon[-1], 1, 1))
    tmp_demand[tmp_demand != 0] = 1
    demand[demand != 0] = 5
    varray1 = calcFun(data.dk,data.dl,args.horizon[-1],demand)
    varray2 = 1.5 * tmp_demand
    data.tmpdir = tempfile.mkdtemp(dir = "/storage/work/dua143/PatSchedPang_v2/FairScheduling/gams_try")
    with open(data.tmpdir+"/params.txt","w") as params_file:
        json.dumps(vars(args), params_file)
    if len(sys.argv) != 1:
        params_file =
        params_file.write("".join(map(str,myopts)))
        params_file.close()
    result_file = tempfile.NamedTemporaryFile(suffix='.csv',
                                              prefix='_'.join(policy)+'testPaperSingleOpt_detLam_result_gen_loc_all_constant_Oct7_hammer',
                                              dir=data.tmpdir,delete=False)
    C_file = tempfile.NamedTemporaryFile(suffix='.csv',
                                              prefix='C_table'+'testPaperSingleOpt_detLam_result_gen_loc_all_constant_Oct7_hammer',
                                              dir=data.tmpdir,delete=False)
    data.dat_C = pd.DataFrame(columns = ['Run','Horizon','Theta','Choice','Policy','Loc_Dep','Joint','Doubly','day','C11','C12','C21','C22','C31','C32','C41','C42','C51','C52'])
    dat_C = data.dat_C.copy()
    c_file = open(C_file.name,'w')
    c_file.write('%s'%','.join(map(str,['Run','Horizon','Theta','Choice','Policy','Loc_Dep','Joint','Doubly','day','C11','C12','C21','C22','C31','C32','C41','C42','C51','C52'])))
    c_file.write('\n')
    with open(result_file.name, 'w') as outfile:
        outfile.write('%s' % ','.join(map(str, ['Run','Horizon','Theta','Choice','Policy','Loc_Dep','Joint','Doubly','Profit'])))
        outfile.write('\n')

        for dob in doubly:
            l_original = calc_lambda(data,dob)
            for data.joint in joint:
                for ch in range(dchoice_a,dchoice_b):
                    for data.dd in args.horizon:
                        data.v = calculate_choice(data.dd, data.dk, data.dt,data.dc, data.dl,ch,[varray1, varray2])
                        data.r = np.array([max(1 - 0.04 * j, 0.6) for j in range(data.dd)])
                        for data.theta in theta:
                            for data.dynamic in policy:
                                for loc_dep in range(loc_a,loc_b):
                                    data.Prob = np.zeros([data.dd, data.dk, data.dt, data.dc, data.dl])
                                    results = None
                                    data.l[1,:,:] = l_original.copy()
                                    data.l[1, :, :] = data.l[1, :, :].sum(axis=0) if loc_dep is 1 else data.l[1, :, :]
                                    # flex_data = float(data.l[1, 0, :]) if dob == 0 else list(data.l[1,0,:])
                                    data.C = Cold.copy()
                                    if data.dynamic == "DP":  # Dynamic
                                        data.Prob = generateDailyProb(data, "SP",loc_dep=loc_dep)
                                        if (data.Prob < 0).any() or (data.Prob > 1).any():
                                            pd.DataFrame(np.array([data.theta, data.dd, ch, data.dynamic])).to_csv("error.csv")
                                        pool = mp.Pool(processes=data.processes)
                                        results = pool.map(functools.partial(run_main, data=data), range(Imax))
                                        # for ii in range(Imax):
                                        #    print run_main(ii, data=data)
                                    elif data.dynamic == "DCPP":
                                        data.Prob = generateDailyProb(data, "CP", loc_dep=loc_dep)
                                        if (data.Prob < 0).any() or (data.Prob > 1).any():
                                            pd.DataFrame(np.array([data.theta, data.dd, ch, data.dynamic])).to_csv("error.csv")
                                        pool = mp.Pool(processes=data.processes)
                                        results = pool.map(functools.partial(run_main, data=data), range(Imax))
                                        # for ii in range(Imax):
                                        #    run_main(ii, data=data)
                                    elif data.dynamic in ["SP","SPP","CP"]:  # Static
                                        data.Prob = generateDailyProb(data, data.dynamic, 1,loc_dep=loc_dep, keepfiles=True)
                                        # break
                                        if (data.Prob < 0).any() or (data.Prob > 1).any():
                                            pd.DataFrame(np.array([data.theta, data.dd, ch, data.dynamic])).to_csv("error.csv")
                                        pool = mp.Pool(processes=data.processes)
                                        results = pool.map(functools.partial(run_main, data=data), range(Imax))
                                        # for ii in range(Imax):
                                        #    print run_main(ii, data=data)

                                    if results is not None:
                                        res = np.array([result[0] for result in results])
                                        dat_C = pd.concat([result[2] for result in results])
                                        dat_C.to_csv(c_file, header=False)
                                        out = pd.DataFrame(np.array(
                                            [np.repeat(data.dd, Imax), np.repeat(data.theta, Imax), np.repeat(ch, Imax),
                                             np.repeat(data.dynamic, Imax),np.repeat(loc_dep, Imax),np.repeat(data.joint, Imax),np.repeat(dob,Imax), res]).T)
                                        out.to_csv(outfile, header=False)
                                        data.l[1,:,:] = np.copy(l_original)
                                #for ii in range(Imax):
                                #    print(run_main(ii, data=data))
    outfile.close()
    c_file.close()
    # pivot_tab = tempfile.NamedTemporaryFile(suffix='.csv',prefix='_'.join(policy) + 'pivot_table',dir=data.tmpdir, delete=False)
    # out.columns = ['Horizon','Theta','Choice','Policy','Loc_Dep','Joint','Doubly','Profit']
    # out["Profit"] = out["Profit"].astype(float)
    # out.pivot_table(values="Profit", index = ['Horizon','Theta','Choice','Loc_Dep','Joint','Doubly'], columns="Policy", aggfunc=[np.mean, np.count_nonzero, np.std]).to_csv("pivot_Table.csv")

