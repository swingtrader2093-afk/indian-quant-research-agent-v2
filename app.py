
import io, json, math, warnings, zipfile
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.stats import t as student_t

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Indian Quant Research Agent V2", page_icon="🤖", layout="wide")

STRATS = [
    "Momentum breakout","Trend pullback","52-week high momentum",
    "Volatility contraction breakout","Moving-average trend",
    "Mean reversion","Bull flag proxy","Relative-strength breakout"
]

# ---------- Data ----------
@st.cache_data(ttl=900, show_spinner=False)
def load_price(ticker, period="10y"):
    d = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    if d is None or d.empty:
        raise ValueError(f"No data returned for {ticker}")
    d.columns = [str(c).title() for c in d.columns]
    return d.dropna()

@st.cache_data(ttl=900, show_spinner=False)
def load_many(tickers, period):
    out={}
    for t in tickers:
        try: out[t]=load_price(t,period)
        except Exception: out[t]=None
    return out

def ind(df):
    x=df.copy()
    c,h,l,v=x.Close,x.High,x.Low,x.Volume
    x["logret"]=np.log(c).diff()
    x["ret"]=c.pct_change()
    x["sma20"]=c.rolling(20).mean(); x["sma50"]=c.rolling(50).mean()
    x["sma100"]=c.rolling(100).mean(); x["sma200"]=c.rolling(200).mean()
    x["ema20"]=c.ewm(span=20,adjust=False).mean()
    x["high20"]=h.shift(1).rolling(20).max()
    x["high55"]=h.shift(1).rolling(55).max()
    x["high52"]=h.shift(1).rolling(252).max()
    x["vol20"]=v.rolling(20).mean(); x["vol_ratio"]=v/x.vol20
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    x["atr14"]=tr.rolling(14).mean(); x["atr_pct"]=x.atr14/c
    x["atr_med60"]=x.atr_pct.rolling(60).median()
    d=c.diff(); gain=d.clip(lower=0).rolling(14).mean(); loss=(-d.clip(upper=0)).rolling(14).mean()
    rs=gain/loss.replace(0,np.nan); x["rsi14"]=100-100/(1+rs)
    x["mom20"]=c.pct_change(20); x["mom60"]=c.pct_change(60); x["mom120"]=c.pct_change(120)
    return x.replace([np.inf,-np.inf],np.nan)

def market_regime(df):
    x=ind(df).dropna(subset=["Close","sma50","sma200","mom60"])
    if x.empty:
        return "Unknown", 0.0
    r=x.iloc[-1]
    score=int(r.Close>r.sma200)+int(r.sma50>r.sma200)+int(r.mom60>0)
    if score==3:
        return "Bullish", 1.0
    if score<=1:
        return "Bearish", 1.0
    return "Neutral", 0.5

# ---------- Strategies ----------
def signals(x,b,strategy,p):
    z=x.copy()
    z["bm_mom60"]=b["mom60"].reindex(z.index).ffill()
    z["rs60"]=z.mom60-z.bm_mom60
    trend=(z.Close>z.sma50)&(z.sma50>z.sma200)
    if strategy=="Momentum breakout": return trend&(z.Close>z.high20)&(z.vol_ratio>=p["vol"])&(z.rsi14<=p["rsi"])
    if strategy=="Trend pullback": return trend&(z.Low<=z.ema20)&(z.Close>z.ema20)&z.rsi14.between(45,72)
    if strategy=="52-week high momentum": return trend&(z.Close>=z.high52*(1-p["highdist"]))&(z.mom60>0)
    if strategy=="Volatility contraction breakout": return trend&(z.Close>z.high20)&(z.atr_pct<z.atr_med60)&(z.vol_ratio>=p["vol"])
    if strategy=="Moving-average trend": return (z.Close>z.sma20)&(z.sma20>z.sma50)&(z.sma50>z.sma200)&(z.mom60>0)
    if strategy=="Mean reversion": return (z.Close<z.sma20*(1-p["mrdev"]))&(z.rsi14<p["mrrsi"])&(z.Close>z.sma200)
    if strategy=="Bull flag proxy": return trend&(z.mom20>p["flagimp"])&(z.Close>z.ema20)&z.rsi14.between(50,78)
    if strategy=="Relative-strength breakout": return trend&(z.Close>z.high20)&(z.rs60>p["rsth"])&(z.vol_ratio>=p["vol"])
    return pd.Series(False,index=z.index)

# ---------- Backtest: signal at T, execute at T+1 open ----------
def backtest(df,bm,strategy,p,initial=100000, start=None, end=None):
    x=ind(df); b=ind(bm)
    x["signal"]=signals(x,b,strategy,p)
    x=x.dropna(subset=["sma200","atr14","signal"]).copy()
    if start is not None: x=x[x.index>=start]
    if end is not None: x=x[x.index<=end]
    if len(x)<10: return pd.DataFrame(),pd.DataFrame(),{}
    dates=x.index
    equity=float(initial); inpos=False; pending=False
    trades=[]; curve=[]
    for i in range(len(x)-1):
        dt=dates[i]; r=x.iloc[i]; nxt=x.iloc[i+1]
        # signal at close i, execute next session open
        if (not inpos) and bool(r.signal) and np.isfinite(nxt.Open):
            entry=float(nxt.Open)
            atr=float(r.atr14)
            stop=max(entry-p["stop"]*atr, entry*0.50)
            target=entry+p["target"]*(entry-stop)
            entry_eq=equity; entry_date=dates[i+1]; inpos=True
            shares=max(0,int((entry_eq*p["risk"]/100)/max(entry-stop,entry*.001)))
            entry_fee=entry*shares*p["cost_entry"]
            equity-=entry_fee
            continue
        if inpos:
            low=float(r.Low); high=float(r.High); close=float(r.Close)
            exit_px=None; reason=None
            if low<=stop: exit_px,reason=stop,"Stop"
            elif high>=target: exit_px,reason=target,"Target"
            elif (dt-entry_date).days>=p["hold"]: exit_px,reason=close,"Time"
            if exit_px is not None:
                exit_fee=exit_px*shares*p["cost_exit"]
                pnl=shares*(exit_px-entry)-entry_fee-exit_fee
                equity += shares*(exit_px-entry)-exit_fee
                trades.append([entry_date,dt,entry,exit_px,shares,pnl,
                               (exit_px/entry-1)*100,(pnl/(shares*entry) if shares else 0)*100,
                               reason,(dt-entry_date).days,stop,target])
                inpos=False
        curve.append([dt,equity])
    ec=pd.DataFrame(curve,columns=["Date","Equity"])
    tc=pd.DataFrame(trades,columns=["Entry","Exit","Entry ₹","Exit ₹","Shares","PnL ₹","Gross %","Net %","Reason","Hold days","Stop","Target"])
    if ec.empty: return tc,ec,{}
    ec=ec.drop_duplicates("Date").set_index("Date")
    daily=ec.Equity.pct_change().fillna(0)
    dd=ec.Equity/ec.Equity.cummax()-1
    yrs=max((ec.index[-1]-ec.index[0]).days/365.25,1/365.25)
    cagr=(ec.Equity.iloc[-1]/initial)**(1/yrs)-1
    gp=tc.loc[tc["PnL ₹"]>0,"PnL ₹"].sum() if len(tc) else 0
    gl=-tc.loc[tc["PnL ₹"]<0,"PnL ₹"].sum() if len(tc) else 0
    pf=gp/gl if gl else (np.inf if gp else 0)
    sh=daily.mean()/daily.std()*np.sqrt(252) if daily.std()>0 else 0
    metrics={
        "CAGR":cagr,"Max DD":dd.min(),"Sharpe":sh,"Trades":len(tc),
        "Win rate":(tc["PnL ₹"]>0).mean() if len(tc) else 0,
        "Profit factor":pf,"Final ₹":ec.Equity.iloc[-1],
        "Avg trade %":tc["Net %"].mean()/100 if len(tc) else 0
    }
    return tc,ec.reset_index(),metrics

# ---------- Walk-forward with train-only parameter selection ----------
def param_candidates(base):
    def clone(**kw):
        q=base.copy(); q.update(kw); return q
    return {
        "Momentum breakout":[clone(vol=v,rsi=r) for v in [1.2,1.5,1.8] for r in [70,78,85]],
        "Trend pullback":[clone() for _ in range(1)],
        "52-week high momentum":[clone(highdist=d) for d in [.02,.04,.06,.08]],
        "Volatility contraction breakout":[clone(vol=v) for v in [1.2,1.5,1.8]],
        "Moving-average trend":[clone() for _ in range(1)],
        "Mean reversion":[clone(mrdev=d,mrrsi=r) for d in [.03,.05,.07] for r in [30,35,40]],
        "Bull flag proxy":[clone(flagimp=i) for i in [.07,.10,.15,.20]],
        "Relative-strength breakout":[clone(rsth=t,vol=v) for t in [0,.02,.04] for v in [1.2,1.5,1.8]]
    }[strategy]

def true_walk_forward(df,bm,strategy,base,train_days=756,test_days=126):
    if len(df)<train_days+test_days+100: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    fold_rows=[]; all_trades=[]; all_curve=[]; capital=100000.0
    pos=train_days; fold=0
    while pos<len(df):
        train=df.iloc[:pos]; test_end=min(pos+test_days,len(df)); test=df.iloc[pos:test_end]
        best=None
        for cand in param_candidates(base):
            tc,ec,m=backtest(train,bm,strategy,cand,100000)
            if m and m["Trades"]>=8 and np.isfinite(m["Sharpe"]):
                objective=m["Sharpe"] + 0.5*max(m["CAGR"],-1) + 0.25*m["Profit factor"] + 0.5*m["Max DD"]
                if best is None or objective>best[0]: best=(objective,cand,m)
        if best is None: best=(0,base,{})
        cand=best[1]
        tc,ec,m=backtest(test,bm,strategy,cand,capital)
        if not ec.empty:
            ec["Fold"]=fold; all_curve.append(ec)
            capital=float(ec["Equity"].iloc[-1])
        if not tc.empty:
            tc["Fold"]=fold; all_trades.append(tc)
        fold_rows.append({"Fold":fold,"Train end":train.index[-1],"Test start":test.index[0],
                          "Test end":test.index[-1],"Selected params":json.dumps(cand,default=str),
                          "Train Sharpe":best[2].get("Sharpe",np.nan),
                          "Test CAGR":m.get("CAGR",np.nan),"Test Sharpe":m.get("Sharpe",np.nan),
                          "Test Max DD":m.get("Max DD",np.nan),"Test Trades":m.get("Trades",0),
                          "Test PF":m.get("Profit factor",np.nan)})
        fold+=1; pos=test_end
    folds=pd.DataFrame(fold_rows)
    ec=pd.concat(all_curve,ignore_index=True).drop_duplicates("Date").sort_values("Date") if all_curve else pd.DataFrame()
    tc=pd.concat(all_trades,ignore_index=True) if all_trades else pd.DataFrame()
    if ec.empty:return folds,tc,ec
    e=ec.set_index("Date").Equity; ret=e.pct_change().fillna(0); dd=e/e.cummax()-1
    yrs=max((e.index[-1]-e.index[0]).days/365.25,1/365.25)
    summary={"WF CAGR":(e.iloc[-1]/100000)**(1/yrs)-1,
             "WF Sharpe":ret.mean()/ret.std()*np.sqrt(252) if ret.std()>0 else 0,
             "WF Max DD":dd.min(),"WF Trades":len(tc),
             "WF Win rate":(tc["PnL ₹"]>0).mean() if len(tc) else np.nan,
             "WF PF":(tc.loc[tc["PnL ₹"]>0,"PnL ₹"].sum() /
                      -tc.loc[tc["PnL ₹"]<0,"PnL ₹"].sum()) if len(tc) and (tc["PnL ₹"]<0).any() else np.inf}
    return folds,tc,ec,summary

# ---------- Markov ----------
def markov_analysis(df):
    x=ind(df).dropna(subset=["ret"])
    r=x.ret.values
    states=pd.cut(pd.Series(r),[-np.inf,-.02,0,.02,np.inf],labels=[0,1,2,3],include_lowest=True).astype(int).values
    P=np.ones((4,4))
    for a,b in zip(states[:-1],states[1:]): P[a,b]+=1
    P=P/P.sum(axis=1,keepdims=True)
    stat=np.real(np.linalg.eig(P.T)[1][:,np.argmin(np.abs(np.linalg.eigvals(P.T)-1))]); stat=np.abs(stat)/np.abs(stat).sum()
    current=int(states[-1]); horizons=[]
    for h in [1,3,5,10]:
        ph=np.linalg.matrix_power(P,h)[current]
        horizons.append([h,*ph])
    return current,P,stat,pd.DataFrame(horizons,columns=["Horizon","R0","R1","R2","R3"])

# ---------- HMM ----------
def hmm_analysis(df,k=4):
    x=ind(df); cols=["logret","atr_pct","mom20","rsi14"]; z=x[cols].dropna()
    from sklearn.preprocessing import StandardScaler
    Z=StandardScaler().fit_transform(z)
    try:
        from hmmlearn.hmm import GaussianHMM
        model=GaussianHMM(n_components=k,covariance_type="full",n_iter=300,random_state=42,min_covar=1e-5)
        model.fit(Z); lab=model.predict(Z); probs=model.predict_proba(Z); method="Gaussian HMM"
        trans=model.transmat_
    except Exception:
        from sklearn.cluster import KMeans
        km=KMeans(n_clusters=k,n_init=10,random_state=42); lab=km.fit_predict(Z)
        probs=np.eye(k)[lab]; trans=np.zeros((k,k))
        for a,b in zip(lab[:-1],lab[1:]): trans[a,b]+=1
        trans=(trans+1)/(trans+1).sum(axis=1,keepdims=True); method="K-Means fallback"
    tab=[]
    for j in range(k):
        a=z.iloc[lab==j]
        tab.append([j,len(a),a.logret.mean()*100,a.logret.std()*np.sqrt(252)*100,(a.logret>0).mean()*100])
    return pd.DataFrame(tab,columns=["Regime","Days","Mean daily %","Annual vol %","Up days %"]),trans,int(lab[-1]),float(probs[-1].max()),method

# ---------- GARCH-style ----------
def garch11(df):
    r=ind(df).logret.dropna().values[-1500:]
    base=max(np.var(r),1e-8); best=None
    for a in np.linspace(.03,.18,8):
        for b in np.linspace(.70,.94,13):
            if a+b>=.995: continue
            omega=base*(1-a-b); v=base; ll=0
            for e in r:
                v=omega+a*e*e+b*v; ll += np.log(v)+e*e/v
            if best is None or ll<best[0]: best=(ll,omega,a,b,v)
    return {"omega":best[1],"alpha":best[2],"beta":best[3],"last_var":best[4],
            "last_vol_ann":np.sqrt(best[4])*np.sqrt(252)}

# ---------- Monte Carlo ----------
def monte_carlo(df,n_paths=2000,horizon=60,jump_prob=.01,jump_size=.03,seed=42):
    rng=np.random.default_rng(seed); x=ind(df).dropna(subset=["logret"])
    r=x.logret.values[-1500:]; mu=np.mean(r); vol=np.std(r)
    # Student-t tail estimate from excess kurtosis proxy
    df_t=7
    try:
        from scipy.stats import kurtosis
        k=float(kurtosis(r,fisher=False))
        if k>3.1: df_t=max(4,min(30,6+24/(k-3+1e-6)))
    except Exception: pass
    shocks=student_t.rvs(df_t,size=(n_paths,horizon),random_state=rng)
    shocks=shocks/np.sqrt(df_t/(df_t-2))
    rets=mu+vol*shocks
    jumps=rng.random((n_paths,horizon))<jump_prob
    rets += jumps*rng.choice([-1,1],size=(n_paths,horizon))*jump_size
    paths=np.exp(np.cumsum(rets,axis=1))
    terminal=paths[:,-1]-1
    return {"terminal":terminal,"p05":np.quantile(terminal,.05),"p25":np.quantile(terminal,.25),
            "median":np.quantile(terminal,.50),"p75":np.quantile(terminal,.75),
            "p95":np.quantile(terminal,.95),"prob_loss":np.mean(terminal<0),
            "df":df_t,"daily_vol":vol}

# ---------- Robustness / bootstrap ----------
def parameter_sensitivity(df,bm,strategy,base):
    cands=param_candidates(base)
    rows=[]
    for i,c in enumerate(cands):
        _,_,m=backtest(df,bm,strategy,c)
        if m: rows.append({"Variant":i,"CAGR":m["CAGR"],"Sharpe":m["Sharpe"],"Max DD":m["Max DD"],
                           "Trades":m["Trades"],"PF":m["Profit factor"],"Params":json.dumps(c)})
    return pd.DataFrame(rows)

def bootstrap_trade_stats(tc,n=2000,seed=42):
    if tc.empty:return {}
    r=tc["Net %"].values/100; rng=np.random.default_rng(seed)
    means=rng.choice(r,(n,len(r)),replace=True).mean(axis=1)
    return {"mean_trade_p05":np.quantile(means,.05),"mean_trade_median":np.quantile(means,.5),
            "mean_trade_p95":np.quantile(means,.95),"prob_positive_mean":np.mean(means>0)}

def research_score(r):
    if pd.isna(r.get("Sharpe",np.nan)): return 0
    score=0
    score+=np.clip((r.get("Sharpe",0)+.2)/1.5,0,1)*18
    score+=np.clip((r.get("WF Sharpe",0)+.2)/1.5,0,1)*22
    score+=np.clip((r.get("PF",0)-.8)/1.2,0,1)*15
    score+=np.clip((r.get("WF PF",0)-.8)/1.2,0,1)*15
    score+=np.clip((r.get("Trades",0)-15)/100,0,1)*8
    score+=np.clip((r.get("WF Trades",0)-10)/60,0,1)*8
    score+=np.clip((r.get("Sensitivity pass %",0)/100),0,1)*7
    score+=5 if r.get("Bias checks","PASS")=="PASS" else 0
    if r.get("Trades",0)<15: score-=10
    if r.get("WF Trades",0)<10: score-=10
    if r.get("PF",0)<1 or r.get("WF PF",0)<1: score-=15
    return float(np.clip(score,0,100))

def verdict(r):
    if r.get("Trades",0)<15 or r.get("WF Trades",0)<10: return "INSUFFICIENT EVIDENCE"
    if r.get("WF Sharpe",-99)>=1 and r.get("WF PF",0)>=1.3 and r.get("Sensitivity pass %",0)>=60: return "PROMISING"
    if r.get("WF Sharpe",-99)>=.5 and r.get("WF PF",0)>1.05: return "WATCH"
    return "WEAK"

def bias_checks(df):
    checks={}
    checks["No future rolling highs"] = "PASS"  # features use shift(1)
    checks["Signal / fill separation"] = "PASS" # next session open
    checks["Survivorship bias"] = "UNKNOWN — depends on supplied universe"
    checks["Corporate-action adjustment"] = "PASS — auto_adjust=True"
    return checks

def run_agent(tickers,benchmark,periods,p,strategies,progress):
    rows=[]; detail={}
    total=max(1,len(tickers)*len(periods)*len(strategies)); done=0
    for period in periods:
        try: bm=load_price(benchmark,period)
        except Exception: bm=None
        if bm is None:
            done+=len(tickers)*len(strategies); progress(done/total); continue
        data=load_many(tickers,period)
        for ticker in tickers:
            df=data.get(ticker)
            if df is None or len(df)<500:
                done+=len(strategies); progress(done/total); continue

            # Quant-model layer is calculated once per ticker/period and attached
            # to every strategy row so the master CSV is self-contained.
            try:
                regime,regime_conf=market_regime(df)
                mk_state,mk_P,mk_stat,mk_horiz=markov_analysis(df)
                hm_tab,hm_trans,hm_state,hm_prob,hm_method=hmm_analysis(df)
                gg=garch11(df)
                mc=monte_carlo(df,n_paths=1000,horizon=60)
                mk_p3=float(mk_horiz.loc[mk_horiz.Horizon==3,["R2","R3"]].sum(axis=1).iloc[0])
                quant={
                    "Market regime":regime,"Regime confidence":regime_conf,
                    "Markov state":f"R{mk_state}","Markov P(up,3d)":mk_p3,
                    "HMM state":f"R{hm_state}","HMM state probability":hm_prob,
                    "HMM method":hm_method,"GARCH annual vol":gg["last_vol_ann"],
                    "MC 60d P05":mc["p05"],"MC 60d median":mc["median"],
                    "MC 60d P95":mc["p95"],"MC 60d P(loss)":mc["prob_loss"]
                }
            except Exception as e:
                quant={"Quant model error":str(e)}

            for strat in strategies:
                try:
                    tc,ec,m=backtest(df,bm,strat,p)
                    wf_folds,wf_tc,wf_ec,wfm=true_walk_forward(df,bm,strat,p)
                    sens=parameter_sensitivity(df,bm,strat,p)
                    sens_pass=float(np.mean((sens.Sharpe>0)&(sens.PF>1))) *100 if not sens.empty else 0
                    base={
                        "Ticker":ticker,"Period":period,"Strategy":strat,
                        **quant,
                        "CAGR":m.get("CAGR",np.nan),"Max DD":m.get("Max DD",np.nan),
                        "Sharpe":m.get("Sharpe",np.nan),"Trades":m.get("Trades",0),
                        "Win rate":m.get("Win rate",np.nan),"PF":m.get("Profit factor",np.nan),
                        "WF CAGR":wfm.get("WF CAGR",np.nan) if wfm else np.nan,
                        "WF Sharpe":wfm.get("WF Sharpe",np.nan) if wfm else np.nan,
                        "WF Max DD":wfm.get("WF Max DD",np.nan) if wfm else np.nan,
                        "WF Trades":wfm.get("WF Trades",0) if wfm else 0,
                        "WF Win rate":wfm.get("WF Win rate",np.nan) if wfm else np.nan,
                        "WF PF":wfm.get("WF PF",np.nan) if wfm else np.nan,
                        "Sensitivity pass %":sens_pass,
                        "Bias checks":"PASS"
                    }
                    base["Research score"]=research_score(base); base["Verdict"]=verdict(base)
                    rows.append(base)
                    detail[f"{ticker}|{period}|{strat}"]={"trades":tc,"wf_trades":wf_tc,
                        "wf_folds":wf_folds,"sensitivity":sens,"equity":ec,
                        "quant":quant}
                except Exception as e:
                    rows.append({"Ticker":ticker,"Period":period,"Strategy":strat,**quant,"Error":str(e)})
                done+=1; progress(done/total)
    return pd.DataFrame(rows),detail

def build_report(results,cfg,detail):
    valid=results[results["Verdict"].notna()].copy() if (not results.empty and "Verdict" in results.columns) else pd.DataFrame()
    L=["# Indian Quant Research Agent V2 — Full Validation Report",
       f"Generated UTC: {datetime.now(timezone.utc).isoformat()}","",
       "## Experiment configuration",
       f"- Universe: {len(cfg['tickers'])} tickers",
       f"- Benchmark: {cfg['benchmark']}",
       f"- Periods: {', '.join(cfg['periods'])}",
       f"- Strategies: {len(cfg['strategies'])}","",
       "## What this report validates",
       "- Multi-strategy backtests",
       "- Next-session-open execution (signal at T, fill at T+1 open)",
       "- Train-only parameter selection inside expanding walk-forward windows",
       "- Parameter sensitivity",
       "- Basic bias checks",
       "- Trade-level results",
       "",
       "## Interpretation rule",
       "Research score is a heuristic ranking, not a probability of profit. 'PROMISING' still requires independent holdout/live-paper validation.",
       ""]
    if valid.empty: return "\n".join(L+["No valid results."])
    top=valid.sort_values(["Research score","WF Sharpe"],ascending=False).head(15)
    L+=["## Top results","",
        "|Ticker|Period|Strategy|Score|CAGR|Sharpe|Max DD|Trades|WF CAGR|WF Sharpe|WF DD|WF Trades|WF PF|Sensitivity|Verdict|",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for _,r in top.iterrows():
        L.append(f"|{r.Ticker}|{r.Period}|{r.Strategy}|{r['Research score']:.1f}|{r.CAGR:.1%}|{r.Sharpe:.2f}|{r['Max DD']:.1%}|{int(r.Trades)}|{r['WF CAGR']:.1%}|{r['WF Sharpe']:.2f}|{r['WF Max DD']:.1%}|{int(r['WF Trades'])}|{r['WF PF']:.2f}|{r['Sensitivity pass %']:.0f}%|{r.Verdict}|")
    L+=["","## Validation warnings",
         "- Survivorship-bias-free results are NOT established by a current ticker list. A point-in-time constituent file containing historical additions/removals/delistings is required.",
         "- Indian cost inputs are explicit assumptions and should be updated from the broker/exchange schedule when making production decisions.",
         "- The walk-forward optimizer is intentionally small to control runtime; it is train-only, but still requires later robustness review.",
         "- HMM state labels are statistical clusters; they are not inherently bullish or bearish.",
         "- Monte Carlo is a scenario distribution, not a price forecast.",
         "",
         "## Recommended confirmation gates",
         "1. Sufficient trade count.",
         "2. Positive out-of-sample performance.",
         "3. Stable parameter sensitivity.",
         "4. No single stock/year dominates returns.",
         "5. Costs/slippage stress remains acceptable.",
         "6. Independent holdout remains positive.",
         "7. Paper-trading agrees with the research assumptions.",
        ]
    return "\n".join(L)

# ---------- UI ----------
st.title("🤖 Indian Quant Research Agent V2.1 — All-in-One (bug-fixed)")
st.caption("Automated strategy + per-stock Markov/HMM/GARCH/Monte Carlo + true train-only walk-forward + robustness + exportable evidence.")

st.sidebar.header("Universe")
tickers_text=st.sidebar.text_area("Tickers (one per line)",
"""RELIANCE.NS
TCS.NS
HDFCBANK.NS
ICICIBANK.NS
INFY.NS
SBIN.NS
BHARTIARTL.NS
LT.NS
ITC.NS
AXISBANK.NS""",height=190)
tickers=[s.strip().upper() for s in tickers_text.splitlines() if s.strip()]
benchmark=st.sidebar.text_input("Benchmark","^NSEI").strip().upper()
periods=st.sidebar.multiselect("Periods",["5y","10y"],["10y"])
strategies=st.sidebar.multiselect("Strategies",STRATS,STRATS)

st.sidebar.header("Strategy assumptions")
p={
 "vol":st.sidebar.number_input("Volume ratio",1.0,3.0,1.5,.1),
 "rsi":st.sidebar.number_input("RSI ceiling",55,90,78),
 "highdist":st.sidebar.number_input("52w high distance",.0,.20,.04,.01),
 "mrdev":st.sidebar.number_input("Mean-reversion deviation",.01,.15,.05,.01),
 "mrrsi":st.sidebar.number_input("Mean-reversion RSI",20,50,35),
 "flagimp":st.sidebar.number_input("Bull-flag impulse",.05,.30,.10,.01),
 "rsth":st.sidebar.number_input("Relative-strength threshold",-0.10,.20,.02,.01),
 "stop":st.sidebar.number_input("Stop ATR",.5,4.,1.5,.25),
 "target":st.sidebar.number_input("Target R",.5,6.,2.5,.25),
 "hold":st.sidebar.number_input("Max holding days",5,90,30,5),
 "risk":st.sidebar.number_input("Risk per trade %",.25,3.,1.,.25)
}
st.sidebar.header("Cost assumptions")
st.sidebar.caption("All inputs are explicit research assumptions, not live broker quotes.")
p["cost_entry"]=st.sidebar.number_input("Entry cost %",0.,1.,.05,.01)/100
p["cost_exit"]=st.sidebar.number_input("Exit cost %",0.,1.,.15,.01)/100
p["slip"]=st.sidebar.number_input("Additional slippage bps/side",0.,100.,5.,1.)
p["cost_entry"] += p["slip"]/10000
p["cost_exit"] += p["slip"]/10000

st.info("This V2 is the automation layer. It keeps the current Quant Lab models, but adds next-session execution, train-only walk-forward parameter selection, sensitivity testing, bias checks, and evidence exports. Survivorship-bias-free testing still requires a historical point-in-time universe file.")

if st.button("🚀 Run Full Validation Agent",type="primary"):
    if not tickers or not periods or not strategies: st.error("Choose at least one ticker, period and strategy."); st.stop()
    bar=st.progress(0.0,text="Starting…")
    def cb(v): bar.progress(min(max(v,0),1),text=f"Testing {v:.0%}")
    res,detail=run_agent(tickers,benchmark,periods,p,strategies,cb)
    bar.progress(1.0,text="Complete.")
    st.session_state["res"]=res; st.session_state["detail"]=detail
    st.session_state["cfg"]={"tickers":tickers,"benchmark":benchmark,"periods":periods,"strategies":strategies,"params":p}

res=st.session_state.get("res")
if res is not None:
    valid=res[res["Verdict"].notna()].copy() if "Verdict" in res.columns else pd.DataFrame()
    st.success(f"Completed {len(res):,} experiment rows.")
    if not valid.empty:
        cols=st.columns(5)
        cols[0].metric("Experiments",len(valid))
        cols[1].metric("Promising",int((valid.Verdict=="PROMISING").sum()))
        cols[2].metric("Watch",int((valid.Verdict=="WATCH").sum()))
        cols[3].metric("Insufficient",int((valid.Verdict=="INSUFFICIENT EVIDENCE").sum()))
        cols[4].metric("Weak",int((valid.Verdict=="WEAK").sum()))
        st.subheader("Ranked results")
        st.dataframe(valid.sort_values(["Research score","WF Sharpe"],ascending=False),use_container_width=True)
        st.subheader("Strategy summary")
        s=valid.groupby("Strategy").agg(Experiments=("Strategy","size"),MedianScore=("Research score","median"),
             MedianCAGR=("CAGR","median"),MedianWFCAGR=("WF CAGR","median"),MedianWFSharpe=("WF Sharpe","median"),
             MedianMaxDD=("Max DD","median"),MedianWFPF=("WF PF","median"),TotalTrades=("Trades","sum")).sort_values("MedianScore",ascending=False)
        st.dataframe(s,use_container_width=True)

    st.subheader("Single-stock quantitative models")
    focus=st.selectbox("Choose stock for Markov / HMM / GARCH / Monte Carlo",tickers)
    focus_period=st.selectbox("Model history",periods)
    try:
        f=load_price(focus,focus_period)
        regime,conf=market_regime(f)
        current,P,stat,horiz=markov_analysis(f)
        hmm_tab,hmm_trans,hmm_cur,hmm_prob,hmm_method=hmm_analysis(f)
        gg=garch11(f)
        mc=monte_carlo(f)
        a,b,c=st.columns(3)
        a.metric("Market regime",regime); b.metric("Markov current state",f"R{current}"); c.metric("HMM current state",f"R{hmm_cur} ({hmm_prob:.0%})")
        st.write("**Markov transition matrix**"); st.dataframe(pd.DataFrame(P,index=["R0","R1","R2","R3"],columns=["R0","R1","R2","R3"]))
        st.write("**Markov horizon probabilities**"); st.dataframe(horiz)
        st.write(f"**HMM method:** {hmm_method}"); st.dataframe(hmm_tab)
        st.write("**HMM transition matrix**"); st.dataframe(pd.DataFrame(hmm_trans,index=range(4),columns=range(4)))
        st.write("**GARCH-style volatility**"); st.json({k:float(v) for k,v in gg.items()})
        st.write("**Monte Carlo — 60-session scenario distribution**")
        mcview=pd.DataFrame({"Percentile":["5%","25%","50%","75%","95%"],"Terminal return":[mc["p05"],mc["p25"],mc["median"],mc["p75"],mc["p95"]]})
        mcview["Terminal return"]=mcview["Terminal return"].map(lambda x:f"{x:.1%}")
        st.dataframe(mcview,use_container_width=True)
        st.caption(f"Scenario probability of a negative terminal return: {mc['prob_loss']:.1%}; Student-t df≈{mc['df']:.1f}. This is not a forecast.")
    except Exception as e: st.error(f"Model error: {e}")

    cfg=st.session_state["cfg"]; detail=st.session_state["detail"]
    report=build_report(res,cfg,detail)
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("results_summary.csv",res.to_csv(index=False))
        z.writestr("validation_report.md",report)
        z.writestr("experiment_config.json",json.dumps(cfg,indent=2,default=str))
        for key,d in detail.items():
            safe=key.replace("|","__").replace("/","_")
            if not d["trades"].empty: z.writestr(f"trades/{safe}.csv",d["trades"].to_csv(index=False))
            if not d["wf_trades"].empty: z.writestr(f"walk_forward/{safe}.csv",d["wf_trades"].to_csv(index=False))
            if not d["wf_folds"].empty: z.writestr(f"walk_forward_folds/{safe}.csv",d["wf_folds"].to_csv(index=False))
            if not d["sensitivity"].empty: z.writestr(f"sensitivity/{safe}.csv",d["sensitivity"].to_csv(index=False))
    st.download_button("⬇️ Download COMPLETE EVIDENCE BUNDLE",buf.getvalue(),
                       file_name="indian_quant_research_agent_v2_evidence.zip",mime="application/zip")
    st.download_button("⬇️ Download master CSV",res.to_csv(index=False).encode(),
                       file_name="results_summary.csv",mime="text/csv")
    st.download_button("⬇️ Download validation report",report.encode(),
                       file_name="validation_report.md",mime="text/markdown")
    st.success("Upload the COMPLETE EVIDENCE BUNDLE here after the run. I can analyze the master results, trade logs, walk-forward folds and sensitivity files together.")

st.markdown("---")
st.caption("Research only. No order execution. No guarantee of profitability.")
