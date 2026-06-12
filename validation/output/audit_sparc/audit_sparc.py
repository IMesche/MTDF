import numpy as np
MRT = "/mnt/e/Projects/projects papers/The Mesche Hypothesis/SSFT PILLAR 01 SPARC Benchmark PASS/massmodels_lelli2016c.mrt"
BETA_KPC = 22685.0; ALPHA = 1.30
Yd, Yb = 0.5, 0.7  # SPARC-standard M/L at 3.6um (disk, bulge) -- frozen, not tuned

gal = {}
for line in open(MRT, encoding="utf-8", errors="ignore"):
    try:
        name=line[0:11].strip()
        R=float(line[19:25]); Vobs=float(line[26:32]); eV=float(line[33:38])
        Vgas=float(line[39:45] or 0); Vdisk=float(line[46:52] or 0); Vbul=float(line[53:59] or 0)
    except: continue
    if not name or Vobs<=0: continue
    gal.setdefault(name,[]).append((R,Vobs,eV,Vgas,Vdisk,Vbul))
gals={k:np.array(v) for k,v in gal.items() if len(v)>=3}
print(f"galaxies parsed: {len(gals)}")

def vbar(a): return np.sqrt(a[:,3]**2 + Yd*a[:,4]**2 + Yb*a[:,5]**2)
def vmtdf(a):
    r=a[:,0]; return vbar(a)*np.sqrt(1+ALPHA/(1+r/BETA_KPC))

print(f"\nSTEP 2: per-galaxy SHAPE fit, fixed M/L {Yd}/{Yb}, v_MTDF=v_bar*sqrt(1+a/(1+r/beta)); enh=sqrt(1+a)={np.sqrt(1+ALPHA):.3f}")
for name in ["NGC3198","DDO154","NGC7331","NGC2403","UGC2885","IC2574"]:
    if name not in gals: print(f"  {name:10s}: not found"); continue
    a=gals[name]; vobs=a[:,1]; eV=np.clip(a[:,2],3.0,None); vm=vmtdf(a)
    chi2=np.sum(((vobs-vm)/eV)**2); nu=len(a)
    print(f"  {name:10s} N={len(a):3d}  chi2/nu={chi2/nu:8.2f}")

print("\nSTEP 3: RAR shape, g=V^2/R; MTDF predicts g_obs=g_bar*(1+a)=const offset at galactic r")
gobs=[]; gbar=[]
for name,a in gals.items():
    r=a[:,0]; vobs=a[:,1]; vb=vbar(a); ok=(r>0)&(vb>0)
    gobs.extend(vobs[ok]**2/r[ok]); gbar.extend(vb[ok]**2/r[ok])
gobs=np.array(gobs); gbar=np.array(gbar)
lo=np.log10(gbar); lg=np.log10(gobs)
slope,inter=np.linalg.lstsq(np.vstack([lo,np.ones_like(lo)]).T,lg,rcond=None)[0]
thr=np.percentile(gbar,33); sel=gbar<thr
sl_lo=np.linalg.lstsq(np.vstack([lo[sel],np.ones(sel.sum())]).T,lg[sel],rcond=None)[0][0]
print(f"  full-sample log-log slope d(log g_obs)/d(log g_bar) = {slope:.3f}")
print(f"  low-acceleration tertile slope                      = {sl_lo:.3f}")
print(f"  observed RAR low-acc slope ~0.5 (McGaugh+2016); MTDF constant-boost => slope ~1.0 (parallel offset)")
print(f"  median g_obs/g_bar = {np.median(gobs/gbar):.3f}  (MTDF predicts ~1+alpha = {1+ALPHA:.2f})")
print("\nVERDICT: slope near 1.0 + per-galaxy chi2/nu>>1 => amplitude reproduced, SHAPE/curvature NOT => re-scope ending.")
