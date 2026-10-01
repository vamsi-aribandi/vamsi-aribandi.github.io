#!/usr/bin/env python3
"""Exact KL/gradient moments and reproducible finite-batch optimization toys.

All samples are on-policy. q is fixed. k1-reward and k3-loss hold sampled
outcomes fixed during differentiation. The optional scalar baseline is the
oracle variance-minimizing action-independent baseline, computed separately
for each gradient estimator. It changes variance, not the expected gradient.
Gradient variance means trace(covariance) in the stated parameterization.

Run experiments (requires NumPy):
  OPENBLAS_NUM_THREADS=1 python tools/kl-gradient-study.py --out /tmp/kl-study
Render figures (also requires Matplotlib):
  python tools/kl-gradient-study.py --mode plots --out /tmp/kl-study
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

ARMS = ['k1_reward', 'k1_reward_baseline', 'k3_loss', 'k3_loss_baseline']
NODES, WEIGHTS = np.polynomial.hermite.hermgauss(32)
WEIGHTS = WEIGHTS / np.sqrt(np.pi)


def normal_expectation(fn, mean, variance):
    return WEIGHTS @ fn(mean + np.sqrt(2 * variance) * NODES)


def summarize(kl, value_vars, g, means, variances, baselines):
    bias2 = ((means - g) ** 2).sum(axis=1)
    return dict(kl=float(kl), value_variance=np.asarray(value_vars).tolist(),
                true_gradient=g.tolist(), gradient_mean=means.tolist(),
                gradient_variance=np.maximum(variances, 0).tolist(),
                gradient_bias_squared=bias2.tolist(), baselines=baselines,
                mse={str(n): (bias2 + variances / n).tolist()
                     for n in [1, 4, 8, 16, 32, 64, 256, 1024]})


def categorical_moments(p, q):
    p, q = np.asarray(p), np.asarray(q)
    score = np.eye(len(p)) - p
    z = np.log(p / q)
    k3 = z + q / p - 1
    norm = (score**2).sum(axis=1)
    coefficients = [z, 1 - q / p]
    gradients, baselines = [], []
    for c in coefficients:
        b = (p * c) @ norm / (p @ norm)
        gradients.extend([c[:, None] * score, (c - b)[:, None] * score])
        baselines.append(float(b))
    means = np.array([p @ v for v in gradients])
    variances = np.array([(p[:, None] * (v - p @ v)**2).sum() for v in gradients])
    g = (p * z) @ score
    assert np.allclose(p @ k3, p @ z)
    assert np.allclose(means[0], g)
    assert np.allclose(means[2], p - q)
    return summarize(p @ z, [p @ (z - p @ z)**2, p @ (k3 - p @ k3)**2],
                     g, means, variances, baselines)


def gaussian_moments(mu, variance, parameters):
    v = variance
    idx = {'mean': [0], 'log_std': [1], 'both': [0, 1]}[parameters]
    def score(x):
        return np.column_stack([(x - mu) / v, (x - mu)**2 / v - 1])[:, idx]
    def z(x):
        return -.5 * np.log(v) - (x - mu)**2 / (2*v) + x*x/2
    def score_norm(x):
        return (score(x)**2).sum(axis=1)
    ep = lambda fn: normal_expectation(fn, mu, v)
    eq = lambda fn: normal_expectation(fn, 0, 1)
    kl = .5 * (mu*mu + v - 1 - np.log(v))
    forward = .5 * ((1 + mu*mu)/v - 1 + np.log(v))
    g = np.array([mu, v - 1])[idx]
    f = np.array([mu/v, 1 - (1 + mu*mu)/v])[idx]
    var1 = mu*mu*v + .5*(v - 1)**2
    sn = ep(score_norm)
    b1 = ep(lambda x: z(x) * score_norm(x)) / sn
    b3 = (sn - eq(score_norm)) / sn
    vg1 = ep(lambda x: z(x)**2 * score_norm(x)) - g @ g
    if v > .5:
        mass = v / np.sqrt(2*v - 1) * np.exp(mu*mu / (2*v - 1))
        tilted = normal_expectation(score_norm, -mu/(2*v - 1), v/(2*v - 1))
        var3 = var1 + mass - 1 - 2*(kl + forward)
        vg3 = sn - 2*eq(score_norm) + mass*tilted - f @ f
    else:
        # q^2/p has a non-integrable Gaussian tail.
        var3, vg3 = np.inf, np.inf
    means = np.array([g, g, f, f])
    variances = np.array([vg1, vg1-b1*b1*sn, vg3, vg3-b3*b3*sn])
    return summarize(kl, [var1, var3], g, means, variances, [float(b1), float(b3)])


def matched_bulge(q, target):
    lo, hi = q[0], 1 - 1e-12
    for _ in range(80):
        a = (lo + hi)/2
        p = np.array([a, (1-a)/2, (1-a)/2])
        if p @ np.log(p/q) < target:
            lo = a
        else:
            hi = a
    return p


def moment_study(out):
    records = []
    for parameters in ['mean', 'log_std', 'both']:
        for mu in [0, .25, .5, 1, 2]:
            for v in [.4, .6, 1, 2, 4]:
                records.append(dict(family='gaussian', parameters=parameters, mu=mu, variance=v,
                                    **gaussian_moments(mu, v, parameters)))
    q = np.array([.1, .45, .45])
    hole = np.array([.001, .4995, .4995])
    cases = dict(hole=hole, shallow_hole=[.05,.475,.475],
                 original_smooth=[.1,.6577534291783038,.2422465708216962],
                 bulge=[.2,.4,.4],
                 matched_bulge=matched_bulge(q, hole @ np.log(hole/q)),
                 asymmetric_bulge=[.22,.43,.35], near_reference=[.11,.445,.445])
    for name, p in cases.items():
        records.append(dict(family='categorical', name=name, p=list(p), q=q.tolist(),
                            parameters='three softmax logits', **categorical_moments(p,q)))
    grid = []
    for q in [np.ones(3)/3, np.array([.1,.45,.45]), np.array([.2,.4,.4]), np.array([.8,.1,.1])]:
        for i in range(1, 40):
            for j in range(1, 40-i):
                p = np.array([i,j,40-i-j])/40
                r = categorical_moments(p,q)
                grid.append(dict(q=q.tolist(), p=p.tolist(), **r))
    # Standard JSON has no Infinity: retain it explicitly as a string.
    def clean(v):
        if isinstance(v, dict): return {k:clean(x) for k,x in v.items()}
        if isinstance(v, list): return [clean(x) for x in v]
        if isinstance(v, (float, np.floating)) and not np.isfinite(v): return 'infinite'
        return v
    (out/'moments.json').write_text(json.dumps(clean(dict(arms=ARMS, named=records, categorical_grid=grid)),indent=2))
    with (out/'moments.csv').open('w') as f:
        writer=csv.writer(f)
        writer.writerow(['family','case','parameters','KL','value_var_k1','value_var_k3',
                         'gradient_var_k1','gradient_var_k1_baseline','gradient_var_k3','gradient_var_k3_baseline','k3_bias_squared'])
        for r in records:
            writer.writerow([r['family'],r.get('name',f"mu={r.get('mu')},v={r.get('variance')}"),r['parameters'],r['kl'],*r['value_variance'],*r['gradient_variance'],r['gradient_bias_squared'][2]])
    for r in records:
        if r['family']=='categorical' or (r['mu'] in [0,.5,1] and r['variance'] in [.6,1,2] and r['parameters']!='both'):
            print({k:r[k] for k in ['family','parameters','value_variance','gradient_variance','gradient_bias_squared']}, {k:r[k] for k in ['name','mu','variance'] if k in r},flush=True)
    for a,b,label in [(0,2,'raw'),(1,3,'both baselines'),(1,2,'k1 baseline vs raw k3')]:
        eligible=[r for r in grid if r['value_variance'][1]<r['value_variance'][0] and r['kl']>1e-10]
        print('GRID',label,'k3 value wins',len(eligible),'of',len(grid),
              'k3 gradient variance loses',sum(r['gradient_variance'][b]>r['gradient_variance'][a] for r in eligible),
              'k3 MSE wins N=4',sum(r['mse']['4'][b]<r['mse']['4'][a] for r in eligible),
              'k3 MSE wins N=256',sum(r['mse']['256'][b]<r['mse']['256'][a] for r in eligible),flush=True)
    return records


def softmax(logits):
    logits=logits-logits.max(axis=-1,keepdims=True)
    logp=logits-np.log(np.exp(logits).sum(axis=-1,keepdims=True))
    return np.exp(logp),logp


def simulate(case, configurations, seeds, seed, budget, batch):
    """Configurations share random numbers; test seeds are separate from tuning."""
    rng=np.random.default_rng(seed)
    rates=np.array([c['lr'] for c in configurations])[:,None]
    use_k3=np.array(['k3' in c['arm'] for c in configurations])[:,None,None]
    use_baseline=np.array(['baseline' in c['arm'] for c in configurations])[:,None]
    count=len(configurations);steps=budget//batch
    failed=np.zeros((count,seeds),dtype=bool)
    history=[];area=np.zeros(count)
    if case['family']=='gaussian':
        initial=np.array([case['mu'],.5*np.log(case['variance'])])
        theta=np.tile(initial,(count,seeds,1))
        idx={'mean':[0],'log_std':[1],'both':[0,1]}[case['parameters']]
        reward=case.get('reward',0.)
    else:
        initial=np.log(case['p']);theta=np.tile(initial,(count,seeds,1))
        q=np.array(case['q']);logq=np.log(q);reward=np.array(case.get('reward',[0.,0.,0.]))
        target,logtarget=softmax(logq+reward)
    for step in range(steps+1):
        with np.errstate(over='ignore',invalid='ignore',divide='ignore',under='ignore'):
            if case['family']=='gaussian':
                bad=(~np.isfinite(theta).all(axis=-1))|(np.abs(theta[...,0])>1e6)|(np.abs(theta[...,1])>30)
                failed|=bad;theta[failed]=initial
                mu=theta[...,0];v=np.exp(2*theta[...,1]);sigma=np.sqrt(v)
                loss=np.zeros_like(mu)
                if 0 in idx:loss+=.5*(mu-reward)**2
                if 1 in idx:loss+=.5*(v-1-np.log(v))
            else:
                theta-=theta.max(axis=-1,keepdims=True)
                bad=(~np.isfinite(theta).all(axis=-1))|(np.ptp(theta,axis=-1)>300)
                failed|=bad;theta[failed]=initial
                p,logp=softmax(theta)
                loss=(p*(logp-logtarget)).sum(axis=-1)
            loss=np.where(failed,np.inf,np.maximum(loss,0))
            if step>0:area+=loss.mean(axis=1)/steps
            if step in {0,1,2,4,8,16,32,64,128,256,512,1024,steps}:
                history.append(dict(samples=step*batch,mean=loss.mean(axis=1).tolist(),
                                    median=np.median(loss,axis=1).tolist(),
                                    p90=np.quantile(loss,.9,axis=1).tolist(),failed=failed.sum(axis=1).tolist()))
            if step==steps:break
            if case['family']=='gaussian':
                eps=rng.normal(size=(1,seeds,batch))
                x=mu[...,None]+sigma[...,None]*eps
                z=.5*(mu[...,None]**2-np.log(v[...,None]))+mu[...,None]*sigma[...,None]*eps+.5*(v[...,None]-1)*eps**2
                # For k3, exp(-z)>exp(600) is marked as numerical divergence.
                bad_ratio=(use_k3 & (-z>600)).any(axis=-1)
                failed|=bad_ratio
                coefficient=np.where(use_k3,-np.expm1(np.minimum(-z,600)),z)-reward*x
                score=np.stack([np.broadcast_to(eps/sigma[...,None],x.shape),np.broadcast_to(eps**2-1,x.shape)],axis=-1)[...,idx]
                sn=np.zeros_like(mu);weighted_z=np.zeros_like(mu);qn=np.zeros_like(mu)
                alpha=.5*(mu*mu-np.log(v));c=.5*(v-1)
                if 0 in idx:
                    sn+=1/v;weighted_z+=(alpha+3*c)/v;qn+=(1+mu*mu)/(v*v)
                if 1 in idx:
                    sn+=2;weighted_z+=2*alpha+10*c;qn+=(3+6*mu*mu+mu**4)/(v*v)-2*(1+mu*mu)/v+1
                baseline=np.where(use_k3[...,0],1-qn/sn,weighted_z/sn)-reward*mu
                coefficient-=np.where(use_baseline,baseline,0)[...,None]
                gradient=(coefficient[...,None]*score).mean(axis=-2)
                gradient=np.where(failed[...,None],0,gradient)
                theta[...,idx]-=rates[...,None]*gradient
            else:
                # Inverse-CDF common random numbers pair each seed across arms/rates.
                uniforms=rng.random((1,seeds,batch))
                actions=(uniforms>p[...,0,None]).astype(int)+(uniforms>(p[...,0]+p[...,1])[...,None]).astype(int)
                counts=np.stack([(actions==i).sum(axis=-1) for i in range(3)],axis=-1)/batch
                coefficients=np.where(use_k3,1-np.exp(logq-logp),logp-logq)-reward
                norm=1-2*p+(p*p).sum(axis=-1,keepdims=True)
                baseline=(p*coefficients*norm).sum(axis=-1)/np.maximum((p*norm).sum(axis=-1),1e-300)
                coefficients-=np.where(use_baseline,baseline,0)[...,None]
                weighted=counts*coefficients
                gradient=weighted-p*weighted.sum(axis=-1,keepdims=True)
                gradient=np.where(failed[...,None],0,gradient)
                theta-=rates[...,None]*gradient
    # Retain per-seed final regrets to compute paired uncertainty after tuning.
    return dict(history=history,area=area.tolist(),final=loss.tolist())


def optimization_study(out,tune_seeds=1024,test_seeds=4096):
    q=[.1,.45,.45]
    cases=[
        dict(name='gaussian_equal_mean',family='gaussian',parameters='mean',mu=.5,variance=1),
        dict(name='gaussian_far_mean',family='gaussian',parameters='mean',mu=2,variance=1),
        dict(name='gaussian_narrow_mean',family='gaussian',parameters='mean',mu=.5,variance=.6),
        dict(name='gaussian_wide_mean',family='gaussian',parameters='mean',mu=.5,variance=2),
        dict(name='gaussian_wide_mean_reward',family='gaussian',parameters='mean',mu=.5,variance=2,reward=.1),
        dict(name='gaussian_narrow_scale',family='gaussian',parameters='log_std',mu=0,variance=.6),
        dict(name='gaussian_wide_scale',family='gaussian',parameters='log_std',mu=0,variance=2),
        dict(name='gaussian_joint',family='gaussian',parameters='both',mu=.5,variance=2),
        dict(name='categorical_hole',family='categorical',p=[.001,.4995,.4995],q=q),
        dict(name='categorical_original_smooth',family='categorical',p=[.1,.6577534291783038,.2422465708216962],q=q),
        dict(name='categorical_bulge',family='categorical',p=[.2,.4,.4],q=q),
        dict(name='categorical_bulge_reward',family='categorical',p=[.2,.4,.4],q=q,reward=[.2,0,0]),
    ]
    results=[]
    for case in cases:
        lrs=[.003,.01,.03,.1,.3,1.,3.]
        if case['family']=='categorical':lrs += [10.,30.,100.]
        if case['name']=='categorical_hole':
            lrs=sorted(set(np.geomspace(.003,100,45).tolist()+[.1,.3,1.,3.,10.,15.,20.,25.,30.]))
        configs=[dict(arm=arm,lr=lr) for arm in ARMS for lr in lrs]
        for batch in [4,32]:
            budget=512
            tuning=simulate(case,configs,tune_seeds,8128,budget,batch)
            final=np.array(tuning['final'])
            selected=[]
            for arm in ARMS:
                indices=[i for i,c in enumerate(configs) if c['arm']==arm]
                # Treat numerical-zero regrets as a tie, then prefer lower AUC.
                best=min(indices,key=lambda i:(max(final[i].mean(),1e-12),tuning['area'][i]))
                selected.append(configs[best])
            test=simulate(case,selected,test_seeds,65537,budget,batch)
            row=dict(case=case,batch=batch,budget=budget,tune_seeds=tune_seeds,test_seeds=test_seeds,
                     selected=selected,tuning_mean_final=final.mean(axis=1).tolist(),tuning_configs=configs,**test)
            results.append(row)
            means=np.array(test['final']).mean(axis=1)
            print(case['name'],'N',batch,'lr',[c['lr'] for c in selected],'regret',means,'fail',test['history'][-1]['failed'],flush=True)
            # Save progress after each complete case; no final-result substitution.
            dump_json(out/'optimization.json',dict(arms=ARMS,results=results))
    return results


def dump_json(path,value):
    def clean(v):
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
        if isinstance(v,list):return [clean(x) for x in v]
        if isinstance(v,(float,np.floating)) and not np.isfinite(v):return 'infinite'
        return v
    path.write_text(json.dumps(clean(value),indent=2))


def summary_plots(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bg,ink,muted,line='#fbf1c7','#3c3836','#75675a','#d5c4a1'
    blue,orange='#076678','#af3a03'
    plt.rcParams.update({'figure.facecolor':bg,'axes.facecolor':bg,'text.color':ink,'axes.labelcolor':muted,
        'xtick.color':muted,'ytick.color':muted,'axes.edgecolor':line,'axes.spines.top':False,'axes.spines.right':False,
        'font.family':'DejaVu Sans','font.size':10})
    data=json.loads((out/'moments.json').read_text())
    named={r['name']:r for r in data['named'] if r['family']=='categorical'}
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    fig.subplots_adjust(left=.075,right=.98,top=.82,bottom=.19,wspace=.33,hspace=.55)
    fig.text(.075,.965,'Same KL. Three different estimator decisions.',fontsize=21,fontfamily='DejaVu Serif')
    fig.text(.075,.91,'q = (0.1, 0.45, 0.45)  ·  KL[p ∥ q] = 0.09965049 in every column',color=muted)
    labels=['Probability hole','Original smooth shift','More mass on the rare action']
    for col,(key,label) in enumerate(zip(['hole','original_smooth','matched_bulge'],labels)):
        r=named[key]
        ax=axes[0,col];ax.bar([0,1],r['value_variance'],color=[blue,orange],width=.55)
        ax.set_yscale('log');ax.set_ylim(.003,20);ax.set_xticks([0,1],['k₁','k₃']);ax.set_ylabel('Variance of the KL estimate')
        ax.set_title(label,loc='left',fontsize=11,pad=16)
        for i,v in enumerate(r['value_variance']):ax.text(i,v*1.18,f'{v:.4g}',ha='center',fontsize=9)
        ax=axes[1,col];batch=np.geomspace(1,1024,200)
        for arm,color,label in [(1,blue,'k₁ in reward'),(3,orange,'k₃ as loss')]:
            variance=r['gradient_variance'][arm];bias=r['gradient_bias_squared'][arm]
            ax.plot(batch,variance/batch+bias,color=color,lw=2.5,label=label)
        ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xlabel('Batch size');ax.set_ylabel('Gradient MSE');ax.set_xticks([1,4,16,64,256,1024],['1','4','16','64','256','1024'])
        ax.grid(axis='y',color=line,lw=.6);ax.legend(frameon=False,fontsize=9)
        if key=='matched_bulge':
            cross=(r['gradient_variance'][1]-r['gradient_variance'][3])/r['gradient_bias_squared'][3]
            ax.axvline(cross,color=muted,lw=1,ls=':');ax.text(cross*1.15,.0012,f'crossover ≈ {cross:.1f}',fontsize=9,color=muted)
    fig.text(.075,.045,'Gradient error is measured against ∇ KL[p ∥ q] in three softmax logits. Both gradient estimators use their own\nvariance-minimizing scalar baseline. All values are exact sums over three actions; these are not training curves.',color=muted,fontsize=10)
    fig.savefig(out/'categorical-tradeoff.png',dpi=180);fig.savefig(out/'categorical-tradeoff.svg');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(10,4.8));fig.subplots_adjust(left=.08,right=.98,top=.76,bottom=.23,wspace=.3)
    fig.text(.08,.94,'Better KL estimates can give noisier unbiased gradients.',fontsize=19,fontfamily='DejaVu Serif')
    fig.text(.08,.86,'p = N(μ, 1), q = N(0, 1)  ·  only μ is learned  ·  both expected gradients equal μ',color=muted)
    mu=np.linspace(.02,1.1,300);t=mu*mu
    axes[0].plot(mu,t,color=blue,lw=2.5,label='k₁');axes[0].plot(mu,np.expm1(t)-t,color=orange,lw=2.5,label='k₃')
    axes[1].plot(mu,2*t+t*t/4,color=blue,lw=2.5,label='k₁ in reward');axes[1].plot(mu,np.exp(t)*(1+4*t)-1-3*t,color=orange,lw=2.5,label='k₃ as loss')
    for ax,ylabel in zip(axes,['Variance of the KL estimate','Variance of the gradient']):
        ax.set_yscale('log');ax.set_xlabel('Policy mean μ');ax.set_ylabel(ylabel);ax.grid(axis='y',color=line,lw=.6);ax.legend(frameon=False,fontsize=10)
    fig.text(.08,.07,'At μ = 0.5: k₃ has 7.3× lower value variance, but 1.6× higher gradient variance.\nThe gradient ranking also holds with an optimal scalar baseline for each estimator.',color=muted,fontsize=10)
    fig.savefig(out/'gaussian-reversal.png',dpi=180);fig.savefig(out/'gaussian-reversal.svg');plt.close(fig)

    results=json.loads((out/'optimization.json').read_text())['results']
    fig,axes=plt.subplots(1,2,figsize=(10,4.8));fig.subplots_adjust(left=.09,right=.98,top=.74,bottom=.24,wspace=.3)
    fig.text(.09,.94,'Lower noise helps—until the objective mismatch matters.',fontsize=18,fontfamily='DejaVu Serif')
    fig.text(.09,.86,'p = N(μ, 2), q = N(0, 1), μ₀ = 0.5  ·  batch size 32  ·  fixed policy variance',color=muted)
    for ax,name,title in zip(axes,['gaussian_wide_mean','gaussian_wide_mean_reward'],['KL only: both objectives favor μ = 0','Reward R(x) = 0.1x: target μ = 0.1']):
        r=next(r for r in results if r['case']['name']==name and r['batch']==32)
        samples=[p['samples'] for p in r['history']]
        for arm,color,label in [(1,blue,'k₁ in reward'),(3,orange,'k₃ as loss')]:
            ax.plot(samples,[p['mean'][arm] for p in r['history']],color=color,lw=2.5,label=label)
        ax.set_yscale('log');ax.set_ylim(.0005,.2);ax.set_xlabel('On-policy samples');ax.set_ylabel('Mean excess target loss');ax.set_title(title,loc='left',fontsize=10,pad=13)
        ax.grid(axis='y',color=line,lw=.6);ax.legend(frameon=False,fontsize=9)
        if name.endswith('reward'):ax.axhline(.005,color=muted,ls=':',lw=1);ax.text(160,.0038,'k₃ equilibrium error = 0.005',fontsize=9,color=muted)
    fig.text(.09,.07,'4,096 held-out paired seeds; learning rates selected separately on 1,024 tuning seeds.\nBoth arms use their own optimal scalar baseline. Target loss is E[−R] + KL[p ∥ q], above its attainable minimum.',color=muted,fontsize=9)
    fig.savefig(out/'optimization-tradeoff.png',dpi=180);fig.savefig(out/'optimization-tradeoff.svg');plt.close(fig)

    with (out/'optimization-summary.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['case','batch','arm','lr','test_seeds','mean_excess_loss','standard_error','median','p90','numerical_failures'])
        for r in results:
            for i,config in enumerate(r['selected']):
                values=np.array([float(v) if v!='infinite' else np.inf for v in r['final'][i]])
                failed=r['history'][-1]['failed'][i]
                writer.writerow([r['case']['name'],r['batch'],config['arm'],config['lr'],r['test_seeds'],values.mean(),values.std(ddof=1)/np.sqrt(len(values)) if not failed else '',np.median(values),np.quantile(values,.9),failed])


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path('/tmp/kl-gradient-study-20261001'))
    parser.add_argument('--mode',choices=['moments','optimization','plots','all'],default='all')
    parser.add_argument('--tune-seeds',type=int,default=1024)
    parser.add_argument('--test-seeds',type=int,default=4096)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if args.mode in ['moments','all']:moment_study(args.out)
    if args.mode in ['optimization','all']:optimization_study(args.out,args.tune_seeds,args.test_seeds)

    if args.mode == 'plots':summary_plots(args.out)
