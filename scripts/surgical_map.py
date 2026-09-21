#!/usr/bin/env python3
"""A 3D map of the operating field from one video, to find the urethra tip (arch apex) in 3D, not per frame.

DA3-L multi-view (Depth Anything 3, ~/pylibs/da3) takes N keyframes of ONE video jointly and returns depth +
camera pose + intrinsics per frame in one shared frame -> fused point cloud. Output `<out>/map.html` is a
self-contained viewer: orbit the cloud, CLICK a point = candidate tip, and every keyframe thumbnail shows where
that 3D point projects (red; an arrow at the edge when it is off-frame). Green = the annotators' consensus tip
in that frame; grey dots in 3D = those consensus tips lifted onto their frame's depth (if the map is
consistent, they cluster). "Download" saves the picked point + its projection into every keyframe.

Foreground is masked (Nick 2026-09-21): ureth_fn segments urethra / prostate / catheter / non-anatomical (robot
arm, instruments); all of it, grown by --dilate px, is dropped from the cloud, so the map is the static
BACKGROUND, where the urethra leaves the pelvic floor. --black-input also blacks it in DA3's input, so the moving
arm cannot pull the poses either (black may read as a flat surface, see CLAUDE.md: compare both).

--window N picks, per video, the N-frame stretch where the annotated tip moves least (camera and tissue most
static). --freeze-first-sight starts the window at the first frame the CATHETER is segmented instead (Nick) (scan every
--sight-stride frames), keeps urethra + prostate ONLY from that frame (still attached to each other) and masks
them in every later keyframe, so the moving / dissected organ is frozen at first sight on the static background.

Cylinder (Nick 2026-09-21): urethra = largest ureth_fn component per keyframe (one segment per frame), eroded,
pooled over all keyframes in 3D -> urethra_cylinder.fit_cylinder (in the middle keyframe's camera frame, radius held
to 0.5-1.5x the silhouette half-width since DA3 units are not mm). END = the annotators' tip (median of the lifted
consensus tips) dropped perpendicular onto the axis; where that perpendicular meets the cylinder surface is the end
(yellow ball + ring). A click (red) is turned into an end point the same way.

Scale is DA3's relative unit, NOT mm (localisation only). The scene must be roughly static across the window:
keep --start/--end to one phase of the dissection.

    python scripts/surgical_map.py --short 749c8234                  # keyframes over the annotated range
    python scripts/surgical_map.py --short 46867a8e --start 300 --end 700 --n 32
    python scripts/surgical_map.py --self-test
"""
import argparse, base64, json, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def backproject(depth, K, c2w):
    """(h,w) depth -> (h,w,3) world points. OpenCV camera (x right, y down, z forward)."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1) * depth[..., None]
    return cam @ c2w[:3, :3].T + c2w[:3, 3]


def project(p, K, w2c):
    """(3,) world point -> (u, v, z_cam)."""
    c = w2c[:3, :3] @ p + w2c[:3, 3]
    return K[0, 0] * c[0] / c[2] + K[0, 2], K[1, 1] * c[1] / c[2] + K[1, 2], c[2]


def as44(e):
    e = np.asarray(e, np.float64)
    return np.vstack([e, [0, 0, 0, 1]]) if e.shape == (3, 4) else e


def seg_model(checkpoint):
    """-> f(RGB uint8) = ureth_fn class map at full res (0 bg, 1 urethra, 2 prostate, 3 catheter, 4 non-anat)."""
    import torch
    from PIL import Image
    from overlay_dir import MetaFormerFPN, predict
    sd = torch.load(checkpoint, map_location="cpu", weights_only=True)
    m = MetaFormerFPN(num_classes=sd["FPN.segmentation_head.0.bias"].shape[0], pretrained="ImageNet",
                      pretrained_weights=None)
    m.load_state_dict(sd)
    m.eval()
    return lambda im: predict(m, Image.fromarray(im), (512, 512), "cpu")   # 512 square = sul_slide_video


def tip_window(rows, n):
    """Start/end of the n-frame window whose annotated tips spread least (mean distance to their median)."""
    f = np.array(sorted(rows)); t = np.array([rows[x]["tip"] for x in f], float)
    best = None
    for s in f[::10]:
        m = (f >= s) & (f < s + n)
        if m.sum() < n // 2:                             # at least half the window annotated
            continue
        spread = np.mean(np.linalg.norm(t[m] - np.median(t[m], 0), axis=1))   # mean: every jump counts
        best = min(best or (np.inf, 0), (spread, int(s)))
    print(f"window: tip spread {best[0]:.0f} px", flush=True)
    return best[1], best[1] + n - 1


def first_sight(fs, seg, stride, frac, cls=3):
    """First frame where class cls (3 = catheter) covers > frac of the frame on two consecutive scans."""
    prev = None
    for k in range(0, len(fs), stride):
        u = (seg(cv2.cvtColor(fs.jpg(k), cv2.COLOR_BGR2RGB)) == cls).mean()
        if u > frac and prev is not None:
            return int(fs.frames[prev])
        prev = k if u > frac else None
    raise SystemExit(f"class {cls} never seen")


def end_point(X, p, d, r):
    """Perpendicular from X to the axis (p, d), continued to the cylinder surface."""
    foot = p + ((X - p) @ d) * d
    return foot + r * (X - foot) / np.linalg.norm(X - foot)


def cylinder(P, w2c):
    """Fit in camera w2c's frame (fit_cylinder assumes the camera at the origin) -> world (p, d, r, t0, t1)."""
    from urethra_cylinder import fit_cylinder, unit
    Pc = P @ w2c[:3, :3].T + w2c[:3, 3]
    c = Pc.mean(0)
    d0 = np.linalg.svd(Pc - c, full_matrices=False)[2][0]
    r_sil = np.subtract(*np.percentile((Pc - c) @ unit(np.cross(d0, unit(c))), [97, 3])) / 2
    p, d, r, res, _ = fit_cylinder(Pc, r_range=(0.5 * r_sil, 1.5 * r_sil))
    R = w2c[:3, :3].T                                    # back to world
    p, d = R @ (p - w2c[:3, 3]), R @ d
    t = (P - p) @ d
    print(f"cylinder: {len(P)} urethra pts, r {r:.4g} (silhouette {r_sil:.4g}), residual {res / r:.2f} r", flush=True)
    return p, d, r, *np.percentile(t, [2, 98])


def build(a):
    import torch
    from arch_tip_data import A, FrameStore
    from depth_anything_3.api import DepthAnything3

    fs = FrameStore(a.short)
    rows = {r["frame"]: r for r in json.loads((A / "labels_all.json").read_text())["rows"] if r["short"] == a.short}
    seg = seg_model(a.checkpoint)
    lo = a.start if a.start is not None else min(rows)
    hi = a.end if a.end is not None else max(rows)
    if a.freeze_first_sight:
        lo = first_sight(fs, seg, a.sight_stride, a.sight_frac)
        hi = lo + (a.window or 400) - 1
        print(f"catheter first seen at frame {lo}", flush=True)
    elif a.window:
        lo, hi = tip_window(rows, a.window)
    hi = min(hi, int(fs.frames[-1]))
    ks = [fs.pos[f] for f in np.unique(np.linspace(lo, hi, a.n).round().astype(int)) if f in fs.pos]
    frames = [int(fs.frames[k]) for k in ks]
    imgs = [cv2.cvtColor(fs.jpg(k), cv2.COLOR_BGR2RGB) for k in ks]
    H0, W0 = imgs[0].shape[:2]
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.dilate + 1,) * 2)
    cls = [int(c) for c in a.mask_classes.split(",")]
    fz = [4]                                             # frozen first-sight frame: only the arm goes
    from overlay_dir import _keep_largest
    segs = [seg(im) for im in imgs]
    fg = [cv2.dilate(np.isin(sg, fz if a.freeze_first_sight and i == 0 else cls).astype(np.uint8), ker) > 0
          for i, sg in enumerate(segs)]
    ero = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))  # urethra edges carry mixed depth
    ure = [cv2.erode((_keep_largest(sg, 1) == 1).astype(np.uint8), ero) for sg in segs]
    print("foreground (dilated) per frame: median %.0f%%" % (100 * np.median([f.mean() for f in fg])), flush=True)
    orig = imgs
    if a.black_input:
        imgs = [np.where(f[..., None], 0, im).astype(np.uint8) for f, im in zip(fg, imgs)]
    print(f"{a.short}: {len(ks)} keyframes {frames[0]}..{frames[-1]}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = DepthAnything3.from_pretrained(a.model).to(dev).eval()
    with torch.no_grad():
        p = m.inference(imgs, process_res=a.res)
    D, C = np.asarray(p.depth, np.float32), np.asarray(p.conf, np.float32)
    Ks, W2C = np.asarray(p.intrinsics, np.float64), np.stack([as44(e) for e in p.extrinsics])
    n, h, w = D.shape
    RGB = np.stack([cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA) for im in orig])  # unblacked colours
    sx, sy = w / W0, h / H0                              # processed = resize of the 1340x1072 crop

    keep = C >= np.percentile(C, a.conf_pct)             # DA3's own GLB export default: drop the lowest 40%
    b = a.margin                                         # frame border = least constrained depth (sul_reveal)
    keep[:, :b], keep[:, -b:], keep[:, :, :b], keep[:, :, -b:] = False, False, False, False
    pts, col, fid, tips3d, ureP = [], [], [], [], []
    for i, k in enumerate(ks):
        c2w = np.linalg.inv(W2C[i])
        gui = cv2.resize((fs.mask(k) | fg[i]).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        RGB[i][gui] //= 3                                # thumbnails show what was masked, dimmed
        ok = keep[i] & ~gui & (D[i] > 0)
        X = backproject(D[i], Ks[i], c2w)
        pts.append(X[ok]); col.append(RGB[i][ok]); fid.append(np.full(ok.sum(), i, np.uint8))
        u8 = cv2.resize(ure[i], (w, h), interpolation=cv2.INTER_NEAREST) > 0
        ureP.append(X[u8 & ok])
        r = rows.get(frames[i])
        if r and r.get("tip"):
            u, v = r["tip"][0] * sx, r["tip"][1] * sy
            if 0 <= u < w and 0 <= v < h and D[i, int(v), int(u)] > 0:   # tip may sit on masked tissue: still lifted
                tips3d.append(X[int(v), int(u)])
    pts, col, fid = np.concatenate(pts), np.concatenate(col), np.concatenate(fid)
    frozen = np.flatnonzero(fid == 0) if a.freeze_first_sight else np.array([], int)  # frozen organ: every point
    rest = np.setdiff1d(np.arange(len(pts)), frozen)
    sel = np.concatenate([frozen, np.random.default_rng(0).choice(
        rest, min(len(rest), max(a.max_points - len(frozen), 0)), replace=False)])
    pts, col, fid = pts[sel], col[sel], fid[sel]         # ponytail: random subsample; voxel grid if it looks noisy

    out = Path(a.out_root) / a.short
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "map.npz", pts=pts, col=col, fid=fid, K=Ks, w2c=W2C, frames=frames,
                        depth=D.astype(np.float16), tips3d=np.array(tips3d).reshape(-1, 3), scale=[sx, sy])
    t = np.array(tips3d).reshape(-1, 3)
    if len(t):
        spread = np.median(np.linalg.norm(t - np.median(t, 0), axis=1)) / np.median(D)
        print(f"annotated tips lifted: {len(t)}, median spread around their median = {spread:.3f} x scene depth")

    cyl = None
    ureP = np.concatenate(ureP)
    if len(ureP) >= 200:
        cp, cd, cr, t0, t1 = cylinder(ureP, W2C[n // 2])
        end = end_point(np.median(t, 0), cp, cd, cr) if len(t) else None
        te = None if end is None else float((end - cp) @ cd)
        cyl = dict(p=cp.tolist(), d=cd.tolist(), r=cr, t0=float(min(t0, te if te is not None else t0)),
                   t1=float(max(t1, te if te is not None else t1)), end=None if end is None else end.tolist())
        print(f"end: t {te} on axis span {t0:.4g}..{t1:.4g}", flush=True)
    else:
        print(f"no cylinder: {len(ureP)} urethra points", flush=True)

    b64 = lambda x: base64.b64encode(np.ascontiguousarray(x).tobytes()).decode()
    thumbs = [base64.b64encode(cv2.imencode(".jpg", cv2.cvtColor(im, cv2.COLOR_RGB2BGR),
                                            [cv2.IMWRITE_JPEG_QUALITY, 80])[1]).decode() for im in RGB]
    cons = [[rows[f]["tip"][0] * sx, rows[f]["tip"][1] * sy] if f in rows and rows[f].get("tip") else None
            for f in frames]
    data = dict(short=a.short, n=len(pts), w=w, h=h, pts=b64(pts.astype(np.float32)), col=b64(col.astype(np.uint8)),
                fid=b64(fid), nf=n, K=Ks.tolist(), w2c=W2C.tolist(), frames=frames, thumbs=thumbs, cons=cons,
                tips3d=t.tolist(), scale=[sx, sy], cyl=cyl)
    (out / "map.html").write_text(HTML.replace("__DATA__", json.dumps(data)), encoding="utf-8")
    print(f"wrote {out / 'map.html'} ({len(pts)} points)")


def self_test():
    rng = np.random.default_rng(0)
    K = np.array([[300., 0, 160], [0, 300, 120], [0, 0, 1]])
    R, _ = cv2.Rodrigues(rng.normal(size=3) * 0.2)
    w2c = np.eye(4); w2c[:3, :3], w2c[:3, 3] = R, [0.1, -0.2, 0.3]
    X = backproject(rng.uniform(1, 3, (240, 320)), K, np.linalg.inv(w2c))
    u, v, z = project(X[100, 200], K, w2c)
    assert abs(u - 200.5) < 1e-6 and abs(v - 100.5) < 1e-6 and z > 0, (u, v, z)
    assert as44(np.eye(4)[:3]).shape == (4, 4)
    rows = {f: {"tip": [500 + (0 if 300 <= f < 700 else rng.normal(0, 60)), 300]} for f in range(1500)}
    assert 290 <= tip_window(rows, 400)[0] <= 310       # the one static stretch wins
    e = end_point(np.array([3., 1, 7]), np.zeros(3), np.array([0., 0, 1]), 2.0)   # axis = z, r 2
    assert np.allclose(e, [2 * 3 / 10 ** .5, 2 / 10 ** .5, 7]), e
    ang = rng.uniform(0, 2 * np.pi, 3000); z = rng.uniform(0, 20, 3000)          # tube r 3 along x, 40 in front
    P = np.stack([z, 3 * np.cos(ang), 40 + 3 * np.sin(ang)], 1)[np.sin(ang) < 0]  # camera-facing half only
    cp, cd, cr, t0, t1 = cylinder(P, np.eye(4))
    assert abs(cr - 3) < 0.3 and abs(abs(cd[0]) - 1) < 1e-2, (cr, cd)
    print("self-test ok")


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>Surgical map</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#111;--fg:#eee;--mut:#999;--pan:#1c1c1c}
body{margin:0;background:var(--bg);color:var(--fg);font:13px system-ui,sans-serif;display:flex;height:100vh}
#v{flex:1;position:relative;min-width:0}#v canvas{display:block}
#side{width:420px;max-width:45vw;overflow:auto;background:var(--pan);padding:10px;box-sizing:border-box}
#hud{position:absolute;left:10px;top:10px;background:#000a;padding:8px;border-radius:6px;line-height:1.7}
.g{display:grid;grid-template-columns:1fr 1fr;gap:6px}.g div{position:relative}
.g canvas{width:100%;display:block;border-radius:3px}.g span{position:absolute;left:4px;top:2px;font-size:11px;
text-shadow:0 0 3px #000}button{background:#333;color:var(--fg);border:1px solid #555;border-radius:4px;padding:3px 8px}
</style></head><body><div id="v"><div id="hud">
<b id="t"></b><br>Drag = orbit, right-drag = pan, wheel = zoom. <b>Click a point = tip.</b><br>
<label><input type="checkbox" id="byf"> colour by keyframe</label>
<label><input type="checkbox" id="tp" checked> annotated tips</label><br>
size <input type="range" id="ps" min="1" max="8" value="2" step="0.5"> <button id="dl">Download tip</button>
<div id="pk" style="color:var(--mut)">no point picked</div></div></div>
<div id="side"><div style="color:var(--mut);margin-bottom:6px">yellow = annotators' end on the cylinder, red = your
end on the cylinder, green = annotators' 2D tip, blue = cylinder axis, arrow = off-frame</div><div class="g" id="g"></div></div>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
"three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const D=__DATA__;const f32=s=>new Float32Array(Uint8Array.from(atob(s),c=>c.charCodeAt(0)).buffer);
const u8=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0));
document.getElementById('t').textContent=D.short+' — '+D.nf+' keyframes, '+D.n.toLocaleString()+' points';
const P=f32(D.pts),C8=u8(D.col),F=u8(D.fid);
const rgb=new Float32Array(P.length),byf=new Float32Array(P.length),c=new THREE.Color();
for(let i=0;i<D.n;i++){for(let j=0;j<3;j++)rgb[3*i+j]=C8[3*i+j]/255;
 c.setHSL(F[i]/D.nf*0.8,0.9,0.55);byf[3*i]=c.r;byf[3*i+1]=c.g;byf[3*i+2]=c.b;}
const el=document.getElementById('v'),R=new THREE.WebGLRenderer({antialias:true});el.appendChild(R.domElement);
const S=new THREE.Scene(),G=new THREE.Group();G.rotation.x=Math.PI;S.add(G);   // OpenCV y-down -> three y-up
const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.BufferAttribute(P,3));
geo.setAttribute('color',new THREE.BufferAttribute(rgb,3));geo.computeBoundingSphere();
const bs=geo.boundingSphere,mat=new THREE.PointsMaterial({size:2,sizeAttenuation:false,vertexColors:true});
G.add(new THREE.Points(geo,mat));
const cam=new THREE.PerspectiveCamera(50,1,bs.radius/1000,bs.radius*20);
const inv=m=>new THREE.Matrix4().fromArray(m.flat()).transpose().invert();       // w2c -> c2w
const c0=new THREE.Vector3().setFromMatrixPosition(inv(D.w2c[0]));G.localToWorld(c0);
const ctr=G.localToWorld(bs.center.clone());                                     // look from frame 0's side, whole field in view
cam.position.copy(ctr).addScaledVector(c0.sub(ctr).normalize(),bs.radius*1.3);cam.lookAt(ctr);
const oc=new OrbitControls(cam,R.domElement);oc.target.copy(ctr);oc.update();
// camera centres
const cc=new Float32Array(3*D.nf);D.w2c.forEach((m,i)=>new THREE.Vector3().setFromMatrixPosition(inv(m)).toArray(cc,3*i));
const cg=new THREE.BufferGeometry();cg.setAttribute('position',new THREE.BufferAttribute(cc,3));
G.add(new THREE.Line(cg,new THREE.LineBasicMaterial({color:0x44aaff})));
const ball=(p,col,r)=>{const m=new THREE.Mesh(new THREE.SphereGeometry(r,12,8),new THREE.MeshBasicMaterial({color:col}));
 m.position.fromArray(p);G.add(m);return m};
const tipObjs=D.tips3d.map(p=>ball(p,0x999999,bs.radius/250));
const pick=ball([0,0,0],0xff2020,bs.radius/120);pick.visible=false;
// urethra cylinder + end points (perpendicular from a point onto the axis, continued to the surface)
const V=a=>new THREE.Vector3().fromArray(a),Cy=D.cyl;
function endOf(x){const p=V(Cy.p),d=V(Cy.d),X=V(x),f=p.clone().addScaledVector(d,X.clone().sub(p).dot(d));
 return f.clone().addScaledVector(X.sub(f).normalize(),Cy.r).toArray()}
function ring(col){const m=new THREE.Mesh(new THREE.TorusGeometry(Cy.r,Cy.r*0.07,8,48),new THREE.MeshBasicMaterial({color:col}));
 m.quaternion.setFromUnitVectors(new THREE.Vector3(0,0,1),V(Cy.d));m.visible=false;G.add(m);return m}
function place(rg,e){const p=V(Cy.p),d=V(Cy.d);rg.position.copy(p).addScaledVector(d,V(e).sub(p).dot(d));rg.visible=true}
let pickRing,yEnd=null;
if(Cy){const L=Cy.t1-Cy.t0,g=new THREE.CylinderGeometry(Cy.r,Cy.r,L,40,1,true),
 m=new THREE.Mesh(g,new THREE.MeshBasicMaterial({color:0x33ccff,transparent:true,opacity:0.22,side:THREE.DoubleSide,depthWrite:false}));
 m.add(new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.CylinderGeometry(Cy.r,Cy.r,L,24,6,true),1),
  new THREE.LineBasicMaterial({color:0x33ccff,transparent:true,opacity:0.35})));
 m.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),V(Cy.d));m.position.copy(V(Cy.p)).addScaledVector(V(Cy.d),(Cy.t0+Cy.t1)/2);
 G.add(m);pickRing=ring(0xff2020);
 if(Cy.end){yEnd=Cy.end;ball(yEnd,0xffd400,bs.radius/120);place(ring(0xffd400),yEnd)}}
// thumbnails
const g=document.getElementById('g'),cv=[];
D.thumbs.forEach((b,i)=>{const d=document.createElement('div'),x=document.createElement('canvas'),s=document.createElement('span');
 x.width=D.w;x.height=D.h;s.textContent='frame '+D.frames[i];d.append(x,s);g.append(d);
 const im=new Image();im.onload=()=>{x.img=im;draw(i)};im.src='data:image/jpeg;base64,'+b;cv.push(x)});
let tip=null;
function proj(p,i){const m=D.w2c[i],K=D.K[i],c=[0,1,2].map(r=>m[r][0]*p[0]+m[r][1]*p[1]+m[r][2]*p[2]+m[r][3]);
 return [K[0][0]*c[0]/c[2]+K[0][2],K[1][1]*c[1]/c[2]+K[1][2],c[2]]}
function mark(x,u,v,col){const k=x.getContext('2d'),r=D.w/45;k.strokeStyle=col;k.lineWidth=D.w/170;
 const inb=u>=0&&v>=0&&u<D.w&&v<D.h;
 if(inb){k.beginPath();k.arc(u,v,r,0,7);k.stroke();k.beginPath();k.moveTo(u-r*1.6,v);k.lineTo(u+r*1.6,v);
  k.moveTo(u,v-r*1.6);k.lineTo(u,v+r*1.6);k.stroke();return}
 const cx=D.w/2,cy=D.h/2,t=Math.min(Math.abs((cx-r)/(u-cx||1e-9)),Math.abs((cy-r)/(v-cy||1e-9))),ex=cx+(u-cx)*t,ey=cy+(v-cy)*t,
 a=Math.atan2(v-cy,u-cx);k.fillStyle=col;k.beginPath();k.moveTo(ex+Math.cos(a)*r,ey+Math.sin(a)*r);
 k.lineTo(ex+Math.cos(a+2.5)*r,ey+Math.sin(a+2.5)*r);k.lineTo(ex+Math.cos(a-2.5)*r,ey+Math.sin(a-2.5)*r);k.fill()}
function draw(i){const x=cv[i];if(!x||!x.img)return;const k=x.getContext('2d');k.drawImage(x.img,0,0);
 if(Cy){const a=proj(V(Cy.p).addScaledVector(V(Cy.d),Cy.t0).toArray(),i),b=proj(V(Cy.p).addScaledVector(V(Cy.d),Cy.t1).toArray(),i);
  if(a[2]>0&&b[2]>0){k.strokeStyle='#3cf';k.lineWidth=D.w/250;k.beginPath();k.moveTo(a[0],a[1]);k.lineTo(b[0],b[1]);k.stroke()}}
 if(D.cons[i])mark(x,D.cons[i][0],D.cons[i][1],'#3f3');
 if(yEnd){const [u,v,z]=proj(yEnd,i);if(z>0)mark(x,u,v,'#fd0')}
 if(tip){const [u,v,z]=proj(tip,i);if(z>0)mark(x,u,v,'#f33')}}
// picking
const rc=new THREE.Raycaster(),mp=new THREE.Vector2();let down;
R.domElement.addEventListener('pointerdown',e=>down=[e.clientX,e.clientY]);
R.domElement.addEventListener('pointerup',e=>{if(!down||Math.hypot(e.clientX-down[0],e.clientY-down[1])>4)return;
 const b=R.domElement.getBoundingClientRect();mp.set((e.clientX-b.left)/b.width*2-1,-(e.clientY-b.top)/b.height*2+1);
 rc.params.Points.threshold=bs.radius/300;rc.setFromCamera(mp,cam);const h=rc.intersectObject(G.children[0])[0];
 if(!h)return;tip=[...P.slice(3*h.index,3*h.index+3)];if(Cy){tip=endOf(tip);place(pickRing,tip)}
 pick.position.fromArray(tip);pick.visible=true;
 document.getElementById('pk').textContent=(Cy?'end on cylinder':'tip')+' = ['+tip.map(v=>v.toFixed(3)).join(', ')+'] (DA3 units)';
 cv.forEach((_,i)=>draw(i))});
document.getElementById('byf').onchange=e=>{geo.setAttribute('color',new THREE.BufferAttribute(e.target.checked?byf:rgb,3))};
document.getElementById('tp').onchange=e=>tipObjs.forEach(o=>o.visible=e.target.checked);
document.getElementById('ps').oninput=e=>mat.size=+e.target.value;
document.getElementById('dl').onclick=()=>{if(!tip)return;const o={short:D.short,tip3d:tip,annotators_end3d:yEnd,cylinder:Cy,
 per_frame:D.frames.map((f,i)=>{const [u,v,z]=proj(tip,i);return {frame:f,x:u/D.scale[0],y:v/D.scale[1],in_front:z>0}})};
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(o,null,1)]));
 a.download=D.short+'_tip.json';a.click()};
function size(){R.setSize(el.clientWidth,el.clientHeight);cam.aspect=el.clientWidth/el.clientHeight;cam.updateProjectionMatrix()}
addEventListener('resize',size);size();R.setAnimationLoop(()=>R.render(S,cam));
</script></body></html>"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--short", default="749c8234")
    ap.add_argument("--start", type=int), ap.add_argument("--end", type=int)
    ap.add_argument("--n", type=int, default=32, help="keyframes, evenly spaced in [start, end]")
    ap.add_argument("--model", default="depth-anything/DA3-LARGE")
    ap.add_argument("--res", type=int, default=504, help="DA3 process_res (long side)")
    ap.add_argument("--conf-pct", type=float, default=40)
    ap.add_argument("--margin", type=int, default=12, help="px trimmed at the processed-frame border")
    ap.add_argument("--max-points", type=int, default=400_000)
    ap.add_argument("--out-root", default="outputs/surgical_map")
    ap.add_argument("--checkpoint", default="outputs/ureth_fn/best.pth")
    ap.add_argument("--dilate", type=int, default=30, help="px (1340x1072 crop) grown around every foreground mask")
    ap.add_argument("--mask-classes", default="1,2,3,4",
                    help="ureth_fn ids to drop: 1 urethra, 2 prostate, 3 catheter, 4 non-anatomical (robot arm)")
    ap.add_argument("--window", type=int, help="frames; pick the stretch where the annotated tip moves least")
    ap.add_argument("--freeze-first-sight", action="store_true",
                    help="window starts at the urethra's first sight; urethra+prostate kept from that frame only")
    ap.add_argument("--sight-stride", type=int, default=10)
    ap.add_argument("--sight-frac", type=float, default=0.005, help="catheter area fraction that counts as seen")
    ap.add_argument("--black-input", action="store_true", help="also black the foreground in DA3's input")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    self_test() if a.self_test else build(a)
