"""
Self-supervised DEPTH fine-tuning of EndoDAC on our RARP videos.

THIS IS DEPTH, NOT SEGMENTATION. Warm-starts from EndoDAC's released depth_model.pth
(NOT the RARP DINO teacher). Trains only LoRA adapters + the depth head's residual/conv
layers (EndoDAC's own recipe) + a small monocular pose net. Loss = Monodepth2-style
photometric reprojection (0.85 SSIM + 0.15 L1) with auto-masking + edge-aware disparity
smoothness, reusing EndoDAC's vendored loss primitives (utils/layers.py).

The deliverable outputs/<run>/best.pth is a plain depth_model state_dict (encoder.* +
depth_head.*, 389 tensors) that drops straight into ATLAS-Interactive's
gui/depth_estimator.py. Run `--self-test` to assert the 389/389 key contract.

Data: RARPAtlas is MONOCULAR 1080p YouTube clips (no stereo) laid out as
  <split>/rarp/<video>/clip_*/images/frame_*.jpg
with consecutive frames per clip -> we build (t-stride, t, t+stride) triplets that have
real camera motion for the photometric loss.

Real run (on Snellius, gpu_h100):
    python scripts/finetune_depth.py \
        --data-root ../data/RARPAtlas \
        --init ../backbones/EndoDAC/depth_model.pth \
        --pose-init-dir ../backbones/EndoDAC \
        --out outputs/rarp_depth --run-name endodac-rarp

ponytail: single-scale Monodepth2 photometric loss (disp upsampled to feed res), mono
pose net only. EndoDAC's full AF-SfMLearner registration/transform/optical-flow refinement
is dropped -- it's a quality enhancement on top of this core, add it back if specular /
non-rigid artifacts show up in the qualitative panels.
"""
from __future__ import annotations
import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

load_dotenv()

# Vendored EndoDAC code (models/ + utils/). Insert at 0 so its `models`/`utils` namespace
# packages win over anything else on the path (mirrors the GUI's sys.path hack).
_ENDODAC = Path(__file__).resolve().parents[1] / "third_party" / "endodac"
sys.path.insert(0, str(_ENDODAC))
os.environ.setdefault("XFORMERS_DISABLED", "1")  # ViT-B uses plain MLP/attention fallbacks

import models.endodac as endodac_pkg          # noqa: E402
import models.backbones as backbones          # noqa: E402
import models.encoders as encoders            # noqa: E402
import models.decoders as decoders            # noqa: E402
from utils.layers import (                    # noqa: E402
    SSIM, BackprojectDepth, Project3D, SpatialTransformer,
    get_occu_mask_backward, get_smooth_loss, get_smooth_bright,
    disp_to_depth, transformation_from_parameters,
)

# EndoDAC / SCARED assumed intrinsics, NORMALISED by image size (fx, fy, cx, cy).
# Override with --intrinsics for real da Vinci values.
DEFAULT_K_NORM = (0.82, 1.02, 0.5, 0.5)

GUI_ARGS = dict(backbone_size="base", r=4, lora_type="dvlora", pretrained_path=None,
                residual_block_indexes=[2, 5, 8, 11], include_cls_token=True)


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def round14(x):
    """ViT-B/14: spatial dims must be multiples of 14."""
    return max(14, int(round(x / 14)) * 14)


# --------------------------------------------------------------------------- data
class RARPTriplets(Dataset):
    """Consecutive-frame triplets from RARPAtlas clips. Returns raw [0,1] color frames
    {-stride,0,+stride}, color-jittered copies for the networks, a vignette valid-mask,
    and intrinsics. No ImageNet norm -- EndoDAC consumes raw [0,1]."""

    def __init__(self, split_dir, hw, k_norm, stride=1, bottom_crop_frac=0.0,
                 augment=False, vignette_thresh=0.04, mask_overlay=True,
                 overlay_frames=16, overlay_std_thresh=6.0, overlay_min_valid=0.25):
        self.h, self.w = hw
        self.stride = stride
        self.bottom_crop_frac = bottom_crop_frac
        self.augment = augment
        self.vignette_thresh = vignette_thresh
        self.to_tensor = transforms.ToTensor()
        # K in pixels for this feed resolution
        fx, fy, cx, cy = k_norm
        K = np.array([[fx * self.w, 0, cx * self.w, 0],
                      [0, fy * self.h, cy * self.h, 0],
                      [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.K = torch.from_numpy(K)
        self.inv_K = torch.from_numpy(np.linalg.pinv(K))

        # index every clip's frames, keep centers that have both neighbours.
        # Per clip, precompute a STATIC-OVERLAY mask: the da Vinci/CMR console GUI
        # (black bars, instrument banner, corner logos/icons) is baked in and identical
        # across frames, while anatomy moves -> low temporal std = overlay. One mask per
        # clip handles every overlay type (incl. bright widgets on no-black-bar frames)
        # without hardcoding geometry. Clips too static to trust fall back to no mask.
        self.samples = []         # (frames[list[Path]], center_idx)
        self.overlay = {}         # str(clip_dir) -> valid mask (1,H,W) float, or None
        clip_dirs = sorted(Path(split_dir).glob("*/*/clip_*/images"))
        for d in clip_dirs:
            frames = sorted(d.glob("frame_*.jpg")) or sorted(d.glob("frame_*.png"))
            for i in range(stride, len(frames) - stride):
                self.samples.append((frames, i))
            self.overlay[str(d)] = self._overlay_mask(
                frames, overlay_frames, overlay_std_thresh, overlay_min_valid) \
                if mask_overlay else None
        assert self.samples, f"no triplets found under {split_dir} (looked for */*/clip_*/images)"

    def _overlay_mask(self, frames, n, thresh, min_valid):
        if len(frames) < 3:
            return None
        idx = np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int)
        stack = np.stack([np.asarray(self._load(frames[i]).convert("L"), np.float32)
                          for i in idx])
        valid = (stack.std(0) > thresh)            # moving = anatomy; static = overlay
        if valid.mean() < min_valid:               # static clip -> temporal signal unreliable
            return None
        return torch.from_numpy(valid[None].astype(np.float32))

    def __len__(self):
        return len(self.samples)

    def _load(self, path):
        img = Image.open(path).convert("RGB")
        if self.bottom_crop_frac > 0:                  # drop baked-in UI banner
            W, H = img.size
            img = img.crop((0, 0, W, int(H * (1 - self.bottom_crop_frac))))
        return img.resize((self.w, self.h), Image.BILINEAR)

    def __getitem__(self, idx):
        frames, c = self.samples[idx]
        offs = {-1: c - self.stride, 0: c, 1: c + self.stride}
        raw = {f: self._load(frames[j]) for f, j in offs.items()}

        if self.augment and random.random() < 0.5:     # same jitter for all 3 (network input)
            cj = transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
            aug = {f: cj(im) for f, im in raw.items()}
        else:
            aug = raw

        out = {}
        for f in (-1, 0, 1):
            out[("color", f)] = self.to_tensor(raw[f])
            out[("color_aug", f)] = self.to_tensor(aug[f])
        # vignette/black-border mask from frame 0 (1 = valid tissue)
        valid = (out[("color", 0)].mean(0, keepdim=True) > self.vignette_thresh).float()
        if valid.mean() < 0.05:                        # frame is mostly black -> trust it all
            valid = torch.ones_like(valid)
        ov = self.overlay.get(str(frames[0].parent))   # AND the static-overlay mask
        if ov is not None:
            valid = valid * ov
        out["valid"] = valid
        out["K"], out["inv_K"] = self.K, self.inv_K
        return out


# ------------------------------------------------------------------------- losses
def reprojection(pred, target, ssim):
    """0.85 SSIM + 0.15 L1, per-pixel (mean over channels)."""
    l1 = (pred - target).abs().mean(1, True)
    s = ssim(pred, target).mean(1, True)
    return 0.85 * s + 0.15 * l1


def smooth_loss_masked(disp, img, valid):
    """Edge-aware smoothness on mean-normalised disp, restricted to valid (non-overlay)
    pixels so the baked GUI banner/bars don't bleed depth into nearby anatomy."""
    nd = disp / (disp.mean([2, 3], keepdim=True) + 1e-7)
    gx = (nd[:, :, :, :-1] - nd[:, :, :, 1:]).abs()
    gy = (nd[:, :, :-1, :] - nd[:, :, 1:, :]).abs()
    igx = (img[:, :, :, :-1] - img[:, :, :, 1:]).abs().mean(1, True)
    igy = (img[:, :, :-1, :] - img[:, :, 1:, :]).abs().mean(1, True)
    gx = gx * torch.exp(-igx) * valid[:, :, :, :-1]
    gy = gy * torch.exp(-igy) * valid[:, :, :-1, :]
    return gx.sum() / valid[:, :, :, :-1].sum().clamp(min=1) \
        + gy.sum() / valid[:, :, :-1, :].sum().clamp(min=1)


def predict_pose(pose_enc, pose_dec, img0, imgf, f):
    """Relative camera transform frame0 -> frame f (Monodepth2 convention)."""
    pair = [imgf, img0] if f < 0 else [img0, imgf]
    axisangle, translation, _ = pose_dec([pose_enc(torch.cat(pair, 1))])
    T = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=(f < 0))
    return T, axisangle[:, 0], translation[:, 0]


# ----------------------------------------- AF-SfMLearner refinement (EndoDAC, optional)
# EndoDAC's full self-supervision: a Position net (dense optical-flow registration +
# occlusion mask) and a Transform net (appearance flow) that builds an illumination-
# corrected "refined" target, so the photometric loss isn't fooled by the specular /
# non-Lambertian lighting changes typical of endoscopy. Two-stage per batch like the
# released trainer: optimise Position alone, then depth+pose+Transform.
def build_refiner(hw, device, init_dir, warm=True):
    R = {
        "pos_enc": encoders.ResnetEncoder(18, False, num_input_images=2).to(device),
        "trans_enc": encoders.ResnetEncoder(18, False, num_input_images=2).to(device),
        "stn": SpatialTransformer(hw).to(device),
        "occu": get_occu_mask_backward(hw).to(device),
    }
    R["pos"] = decoders.PositionDecoder(R["pos_enc"].num_ch_enc, scales=range(4)).to(device)
    R["trans"] = decoders.TransformDecoder(R["trans_enc"].num_ch_enc, scales=range(4)).to(device)
    if warm:
        d = Path(init_dir)
        for k, fn in [("pos_enc", "position_encoder.pth"), ("pos", "position.pth"),
                      ("trans_enc", "transform_encoder.pth"), ("trans", "transform.pth")]:
            if (d / fn).exists():
                _filter_load(R[k], d / fn, k)
    return R


def _refine_predict(R, aug, color, f, hw):
    """frame f -> 0: dense registration (warp), occlusion mask, appearance-refined target."""
    pos = R["pos"](R["pos_enc"](torch.cat([aug[f], aug[0]], 1)))[("position", 0)]
    pos_r = R["pos"](R["pos_enc"](torch.cat([aug[0], aug[f]], 1)))[("position", 0)]
    pos = F.interpolate(pos, hw, mode="bilinear", align_corners=True)
    pos_r = F.interpolate(pos_r, hw, mode="bilinear", align_corners=True)
    registration = R["stn"](color[f], pos)
    occu, _ = R["occu"](pos_r)
    tr = R["trans"](R["trans_enc"](torch.cat([registration, color[0]], 1)))[("transform", 0)]
    tr = F.interpolate(tr, hw, mode="bilinear", align_corners=True)
    refined = torch.clamp(tr * occu.detach() + color[0], 0.0, 1.0)
    return dict(position=pos, registration=registration, occu=occu, transform=tr, refined=refined)


def position_loss(R, batch, hw, ssim, pos_smooth_w):
    """Stage 0: train the Position net only (registration + flow smoothness)."""
    aug = {f: batch[("color_aug", f)] for f in (-1, 0, 1)}
    color = {f: batch[("color", f)] for f in (-1, 0, 1)}
    valid = batch["valid"]
    reg, smooth = 0.0, 0.0
    for f in (-1, 1):
        p = _refine_predict(R, aug, color, f, hw)
        occu = p["occu"] * valid
        reg = reg + (reprojection(p["registration"], p["refined"].detach(), ssim)
                     * occu).sum() / occu.sum().clamp(min=1)
        smooth = smooth + get_smooth_loss(p["position"], color[0])
    return reg / 2.0 + pos_smooth_w * (smooth / 2.0)


def refine_depth_step(batch, depth_model, pose_enc, pose_dec, R, ssim, backproj, project,
                      hw, min_depth, max_depth, w):
    """Stage 1: depth + pose + Transform against the appearance-refined target."""
    aug = {f: batch[("color_aug", f)] for f in (-1, 0, 1)}
    color = {f: batch[("color", f)] for f in (-1, 0, 1)}
    K, inv_K, valid = batch["K"], batch["inv_K"], batch["valid"]
    disp = F.interpolate(depth_model(aug[0])[("disp", 0)], hw, mode="bilinear", align_corners=False)
    _, depth = disp_to_depth(disp, min_depth, max_depth)
    reproj, transf, cvt, tstats = 0.0, 0.0, 0.0, []
    for f in (-1, 1):
        p = _refine_predict(R, aug, color, f, hw)
        occu = (p["occu"] * valid).detach()
        ax, tr, _ = pose_dec([pose_enc(torch.cat([aug[f], aug[0]], 1))])  # EndoDAC: [f,0]
        ax, tr = ax[:, 0], tr[:, 0]
        tstats.append((ax.norm(dim=-1).mean(), tr.norm(dim=-1).mean()))
        T = transformation_from_parameters(ax, tr)
        pix = project(backproj(depth, inv_K), K, T)
        warped = F.grid_sample(color[f], pix, padding_mode="border", align_corners=True)
        reproj = reproj + (reprojection(warped, p["refined"], ssim)
                           * occu).sum() / occu.sum().clamp(min=1)
        transf = transf + ((p["refined"] - p["registration"].detach()).abs().mean(1, True)
                           * occu).sum() / occu.sum().clamp(min=1)
        cvt = cvt + get_smooth_bright(p["transform"], color[0], p["registration"].detach(), occu)
    smooth = smooth_loss_masked(disp, color[0], valid)
    loss = reproj / 2.0 + w["tc"] * (transf / 2.0) + w["ts"] * (cvt / 2.0) + w["ds"] * smooth
    rot = torch.stack([t[0] for t in tstats]).mean()
    trn = torch.stack([t[1] for t in tstats]).mean()
    logs = dict(loss=loss.item(), photo=(reproj / 2).item(), smooth=smooth.item(),
                transform=(transf / 2).item(), pose_rot=rot.item(), pose_trans=trn.item())
    return loss, logs


def _set_stage(R, pose_enc, pose_dec, depth_model, stage):
    """stage 0 = train Position only; stage 1 = train depth(LoRA)+pose+Transform."""
    for p in list(R["pos_enc"].parameters()) + list(R["pos"].parameters()):
        p.requires_grad = (stage == 0)
    for p in (list(R["trans_enc"].parameters()) + list(R["trans"].parameters())
              + list(pose_enc.parameters()) + list(pose_dec.parameters())):
        p.requires_grad = (stage == 1)
    for n, p in depth_model.named_parameters():
        p.requires_grad = (stage == 1) and any(k in n for k in ("lora_", "residual_", "conv_depth_"))
    for m, on in [(R["pos_enc"], stage == 0), (R["pos"], stage == 0),
                  (R["trans_enc"], stage == 1), (R["trans"], stage == 1),
                  (pose_enc, stage == 1), (pose_dec, stage == 1), (depth_model, stage == 1)]:
        m.train(on)


def photometric_step(batch, depth_model, pose_enc, pose_dec, ssim, backproj, project,
                     hw, min_depth, max_depth, smooth_w, automask=True):
    """Returns (loss, logs). Single-scale Monodepth2 photometric + smoothness."""
    h, w = hw
    color = {f: batch[("color", f)] for f in (-1, 0, 1)}
    aug = {f: batch[("color_aug", f)] for f in (-1, 0, 1)}
    K, inv_K, valid = batch["K"], batch["inv_K"], batch["valid"]

    disp = depth_model(aug[0])[("disp", 0)]
    disp = F.interpolate(disp, hw, mode="bilinear", align_corners=False)
    _, depth = disp_to_depth(disp, min_depth, max_depth)

    reproj, ident, tstats = [], [], []
    for f in (-1, 1):
        T, ax, tr = predict_pose(pose_enc, pose_dec, aug[0], aug[f], f)
        tstats.append((ax.norm(dim=-1).mean(), tr.norm(dim=-1).mean()))
        cam_pts = backproj(depth, inv_K)
        pix = project(cam_pts, K, T)
        warped = F.grid_sample(color[f], pix, padding_mode="border", align_corners=True)
        reproj.append(reprojection(warped, color[0], ssim))
        ident.append(reprojection(color[f], color[0], ssim))   # identity = static-pixel baseline

    reproj = torch.cat(reproj, 1).min(1, True)[0]
    if automask:
        ident = torch.cat(ident, 1).min(1, True)[0]
        ident += 1e-5 * torch.randn_like(ident)                # break ties
        combined = torch.min(reproj, ident)
    else:
        combined = reproj

    m = valid
    photo = (combined * m).sum() / m.sum().clamp(min=1)
    smooth = smooth_loss_masked(disp, color[0], valid)
    loss = photo + smooth_w * smooth

    rot = torch.stack([t[0] for t in tstats]).mean()
    trans = torch.stack([t[1] for t in tstats]).mean()
    logs = dict(loss=loss.item(), photo=photo.item(), smooth=smooth.item(),
                pose_rot=rot.item(), pose_trans=trans.item())
    return loss, logs


# ---------------------------------------------------------------------- warm-start
def _filter_load(module, ckpt_path, name):
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    mdict = module.state_dict()
    keep = {k: v for k, v in sd.items() if k in mdict and v.shape == mdict[k].shape}
    msg = module.load_state_dict(keep, strict=False)
    print(f"[warm-start:{name}] loaded {len(keep)}/{len(mdict)} "
          f"(missing={len(msg.missing_keys)})", flush=True)


def make_endodac(image_shape):
    """Build endodac with the EXACT GUI args. The residual blocks are pinned to the
    backbone's input_size grid, so we inject input_size=image_shape into vit_base the
    same way ATLAS's gui/depth_estimator.py does -- otherwise forward() reshapes to the
    wrong (224,280) patch grid at any other resolution."""
    orig = backbones.vits.vit_base
    backbones.vits.vit_base = lambda **kw: orig(input_size=tuple(image_shape), **kw)
    try:
        return endodac_pkg.endodac(image_shape=tuple(image_shape), **GUI_ARGS)
    finally:
        backbones.vits.vit_base = orig


def build_depth_model(image_shape, device):
    model = make_endodac(image_shape).to(device)
    # EndoDAC recipe: train LoRA adapters + encoder residual blocks + depth conv heads.
    for n, p in model.named_parameters():
        p.requires_grad = any(k in n for k in ("lora_", "residual_", "conv_depth_"))
    return model


# ----------------------------------------------------------------------- qualitative
def colorize(disp, valid=None):
    """disp HxW float -> magma uint8 HxW3, normalised over valid region (GUI-style)."""
    import matplotlib
    if valid is None:
        valid = np.ones_like(disp, bool)
    lo, hi = disp[valid].min(), np.percentile(disp[valid], 95)
    out = np.clip((disp - lo) / (hi - lo + 1e-8), 0, 1)
    out[~valid] = 0
    magma = matplotlib.colormaps["magma"]                 # mpl>=3.6 (cm.get_cmap removed in 3.11)
    return (magma(out)[:, :, :3] * 255).astype(np.uint8)


@torch.no_grad()
def qualitative_panel(depth_model, panel, hw, device, min_depth, max_depth):
    """panel: list of (rgb_tensor[3,H,W], valid[1,H,W]). Returns one stacked uint8 image
    [rgb | depth] per row -> wandb.Image-able array."""
    depth_model.eval()
    rows = []
    for rgb, valid in panel:
        x = rgb.unsqueeze(0).to(device)
        disp = depth_model(x)[("disp", 0)]
        disp = F.interpolate(disp, hw, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        v = valid[0].cpu().numpy() > 0.5
        rgb_np = (rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        rows.append(np.concatenate([rgb_np, colorize(disp, v)], axis=1))
    return np.concatenate(rows, axis=0)


# ---------------------------------------------------------------------------- train
def run_epoch(loader, depth_model, pose_enc, pose_dec, ssim, backproj, project, opt,
              device, args, hw, train=True):
    # fp32 (no AMP): the pose net + grid_sample geometry overflow under fp16 autocast and
    # send the whole run to NaN -- EndoDAC itself trains in fp32. Grad-clip for good measure.
    depth_model.train(train); pose_enc.train(train); pose_dec.train(train)
    agg = {}; nb = 0; skipped = 0
    for batch in loader:
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        with torch.set_grad_enabled(train):
            loss, logs = photometric_step(
                batch, depth_model, pose_enc, pose_dec, ssim, backproj, project, hw,
                args.min_depth, args.max_depth, args.smoothness, automask=not args.no_automask)
        if train:
            if not torch.isfinite(loss):           # guard: never step on a NaN/Inf batch
                skipped += 1
                continue
            opt.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in opt.param_groups for p in g["params"]], args.grad_clip)
            opt.step()
        for k, v in logs.items():
            agg[k] = agg.get(k, 0.0) + v
        nb += 1
    if skipped:
        print(f"[warn] skipped {skipped} non-finite batches", flush=True)
    return {k: v / max(nb, 1) for k, v in agg.items()}


def _clip(opt, max_norm):
    if max_norm > 0:
        torch.nn.utils.clip_grad_norm_(
            [p for g in opt.param_groups for p in g["params"]], max_norm)


def run_epoch_refine(loader, depth_model, pose_enc, pose_dec, R, opt, opt0, ssim,
                     backproj, project, device, args, hw, w, train=True):
    """AF-SfMLearner two-stage epoch: optimise Position (opt0), then depth+pose+Transform (opt)."""
    agg = {}; nb = 0; skipped = 0
    for batch in loader:
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        if train:
            _set_stage(R, pose_enc, pose_dec, depth_model, 0)          # Position stage
            loss0 = position_loss(R, batch, hw, ssim, w["ps"])
            if torch.isfinite(loss0):
                opt0.zero_grad(); loss0.backward(); _clip(opt0, args.grad_clip); opt0.step()
            _set_stage(R, pose_enc, pose_dec, depth_model, 1)          # depth/pose/transform
            loss, logs = refine_depth_step(batch, depth_model, pose_enc, pose_dec, R, ssim,
                                           backproj, project, hw, args.min_depth, args.max_depth, w)
            if not torch.isfinite(loss):
                skipped += 1; continue
            opt.zero_grad(); loss.backward(); _clip(opt, args.grad_clip); opt.step()
        else:
            for m in (R["pos_enc"], R["pos"], R["trans_enc"], R["trans"],
                      pose_enc, pose_dec, depth_model):
                m.eval()
            with torch.no_grad():
                loss, logs = refine_depth_step(batch, depth_model, pose_enc, pose_dec, R, ssim,
                                               backproj, project, hw, args.min_depth, args.max_depth, w)
        for k, v in logs.items():
            agg[k] = agg.get(k, 0.0) + v
        nb += 1
    if skipped:
        print(f"[warn] skipped {skipped} non-finite batches", flush=True)
    return {k: v / max(nb, 1) for k, v in agg.items()}


def self_test(args, device):
    """Rebuild with the exact GUI args, save+reload a state_dict, assert the key contract."""
    h, w = (round14(args.image_shape[0]), round14(args.image_shape[1]))
    model = make_endodac((h, w))
    n_keys = len(model.state_dict())
    print(f"[self-test] built endodac{(h, w)} -> {n_keys} tensor keys")

    ckpt = args.ckpt
    if ckpt and Path(ckpt).exists():
        sd = torch.load(ckpt, map_location="cpu")
        src = f"checkpoint {ckpt}"
    else:
        tmp = Path(args.out) / "_selftest.pth"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), tmp)
        sd = torch.load(tmp, map_location="cpu"); tmp.unlink()
        src = "fresh state_dict round-trip"

    mdict = model.state_dict()
    # mirror the GUI loader: keep only keys the model wants (drops height/width/use_stereo)
    keep = {k: v for k, v in sd.items() if k in mdict}
    msg = model.load_state_dict(keep, strict=False)
    matched = len(keep)
    assert matched == n_keys, f"{matched}/{n_keys} keys matched ({src})"
    assert not msg.missing_keys, f"missing keys: {msg.missing_keys[:5]}"
    print(f"[self-test] OK: {matched}/{n_keys} keys load from {src}, 0 missing "
          f"-> GUI-compatible")
    if n_keys != 389:
        print(f"[self-test] WARNING: expected 389 keys, got {n_keys} "
              f"(check backbone/lora args)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="../data/RARPAtlas")
    ap.add_argument("--init", default="../backbones/EndoDAC/depth_model.pth",
                    help="EndoDAC released depth_model.pth to warm-start from")
    ap.add_argument("--pose-init-dir", default="../backbones/EndoDAC",
                    help="dir with pose.pth + pose_encoder.pth (optional warm-start)")
    ap.add_argument("--out", default="outputs/rarp_depth")
    ap.add_argument("--run-name", default="endodac-rarp")
    ap.add_argument("--image-shape", type=int, nargs=2, default=[392, 490],
                    help="train/inference res (H W), multiples of 14. Recorded for the GUI.")
    ap.add_argument("--intrinsics", type=float, nargs=4, default=list(DEFAULT_K_NORM),
                    metavar=("fx", "fy", "cx", "cy"),
                    help="NORMALISED da Vinci intrinsics; default = EndoDAC/SCARED assumed K")
    ap.add_argument("--frame-stride", type=int, default=1, help="triplet baseline in frames")
    ap.add_argument("--bottom-crop-frac", type=float, default=0.0,
                    help="crop this fraction off the bottom (UI banner) before resize")
    ap.add_argument("--no-overlay-mask", action="store_true",
                    help="disable per-clip temporal static-overlay (console GUI) masking")
    ap.add_argument("--overlay-frames", type=int, default=16,
                    help="frames per clip used to estimate the static-overlay mask")
    ap.add_argument("--overlay-std-thresh", type=float, default=6.0,
                    help="temporal std (0-255) below which a pixel is treated as overlay")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0, help="max grad norm; 0 disables")
    ap.add_argument("--smoothness", type=float, default=1e-3)
    ap.add_argument("--refine", action=argparse.BooleanOptionalAction, default=True,
                    help="EndoDAC AF-SfMLearner refinement (Position+Transform nets); "
                         "--no-refine for the plain Monodepth2 path")
    ap.add_argument("--position-smoothness", type=float, default=1e-3)
    ap.add_argument("--transform-constraint", type=float, default=0.01)
    ap.add_argument("--transform-smoothness", type=float, default=0.01)
    ap.add_argument("--min-depth", type=float, default=0.1)
    ap.add_argument("--max-depth", type=float, default=150.0)
    ap.add_argument("--no-automask", action="store_true")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--panel-size", type=int, default=8, help="fixed qualitative frames")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--self-test", action="store_true",
                    help="assert the 389-key GUI contract and exit")
    ap.add_argument("--ckpt", default=None, help="checkpoint for --self-test (default: fresh)")
    ap.add_argument("--smoke", action="store_true", help="tiny synthetic fwd/bwd, no data")
    args = ap.parse_args()
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hw = (round14(args.image_shape[0]), round14(args.image_shape[1]))

    if args.self_test:
        self_test(args, device); return

    if args.smoke:
        m = build_depth_model(hw, device)
        pe = encoders.ResnetEncoder(18, False, num_input_images=2).to(device)
        pd = decoders.PoseDecoder(pe.num_ch_enc, 1, num_frames_to_predict_for=2).to(device)
        ssim = SSIM().to(device)
        bp = BackprojectDepth(2, *hw).to(device); pr = Project3D(2, *hw).to(device)
        fx, fy, cx, cy = args.intrinsics
        K = torch.tensor([[fx * hw[1], 0, cx * hw[1], 0], [0, fy * hw[0], cy * hw[0], 0],
                          [0, 0, 1, 0], [0, 0, 0, 1]]).float().repeat(2, 1, 1).to(device)
        batch = {("color", f): torch.rand(2, 3, *hw, device=device) for f in (-1, 0, 1)}
        batch.update({("color_aug", f): torch.rand(2, 3, *hw, device=device) for f in (-1, 0, 1)})
        batch.update(valid=torch.ones(2, 1, *hw, device=device), K=K, inv_K=torch.inverse(K))
        if args.refine:
            R = build_refiner(hw, device, args.pose_init_dir, warm=False)
            w = dict(ps=args.position_smoothness, tc=args.transform_constraint,
                     ts=args.transform_smoothness, ds=args.smoothness)
            position_loss(R, batch, hw, ssim, w["ps"]).backward()
            loss, logs = refine_depth_step(batch, m, pe, pd, R, ssim, bp, pr, hw,
                                           args.min_depth, args.max_depth, w)
        else:
            loss, logs = photometric_step(batch, m, pe, pd, ssim, bp, pr, hw,
                                          args.min_depth, args.max_depth, args.smoothness)
        loss.backward()
        print(f"[smoke] ok | hw={hw} refine={args.refine} loss={logs['loss']:.4f} device={device}")
        return

    root = Path(args.data_root)
    k_norm = tuple(args.intrinsics)
    ds_kw = dict(mask_overlay=not args.no_overlay_mask, overlay_frames=args.overlay_frames,
                 overlay_std_thresh=args.overlay_std_thresh)
    print(f"[setup] hw={hw} K_norm={k_norm} stride={args.frame_stride} "
          f"overlay_mask={not args.no_overlay_mask} device={device}", flush=True)

    tr = DataLoader(
        RARPTriplets(root / "Train", hw, k_norm, args.frame_stride, args.bottom_crop_frac,
                     augment=not args.no_augment, **ds_kw),
        args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    va_ds = RARPTriplets(root / "Validation", hw, k_norm, args.frame_stride,
                         args.bottom_crop_frac, **ds_kw)
    va = DataLoader(va_ds, args.batch_size, shuffle=False, num_workers=args.workers,
                    pin_memory=True, drop_last=True)

    # fixed qualitative panel (deterministic frames from Validation)
    panel_idx = np.linspace(0, len(va_ds) - 1, args.panel_size).astype(int)
    panel = [(va_ds[i][("color", 0)], va_ds[i]["valid"]) for i in panel_idx]

    depth_model = build_depth_model(hw, device)
    _filter_load(depth_model, args.init, "depth")
    pose_enc = encoders.ResnetEncoder(18, False, num_input_images=2).to(device)
    pose_dec = decoders.PoseDecoder(pose_enc.num_ch_enc, 1, num_frames_to_predict_for=2).to(device)
    pid = Path(args.pose_init_dir)
    if (pid / "pose_encoder.pth").exists():
        _filter_load(pose_enc, pid / "pose_encoder.pth", "pose_enc")
        _filter_load(pose_dec, pid / "pose.pth", "pose")

    R = opt0 = None
    weights = dict(ps=args.position_smoothness, tc=args.transform_constraint,
                   ts=args.transform_smoothness, ds=args.smoothness)
    if args.refine:
        R = build_refiner(hw, device, args.pose_init_dir)
        opt0 = torch.optim.Adam(list(R["pos"].parameters()) + list(R["pos_enc"].parameters()), lr=1e-4)
        depth_train = [p for p in depth_model.parameters() if p.requires_grad]
        params = depth_train + list(pose_enc.parameters()) + list(pose_dec.parameters()) \
            + list(R["trans"].parameters()) + list(R["trans_enc"].parameters())
    else:
        params = [p for p in depth_model.parameters() if p.requires_grad] \
            + list(pose_enc.parameters()) + list(pose_dec.parameters())
    n_train = sum(p.numel() for p in params)
    print(f"[optim] Adam lr={args.lr} | refine={args.refine} | "
          f"stage1 trainable params={n_train/1e6:.2f}M", flush=True)
    opt = torch.optim.Adam(params, lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, args.epochs // 2), gamma=0.1)
    ssim = SSIM().to(device)
    backproj = BackprojectDepth(args.batch_size, *hw).to(device)
    project = Project3D(args.batch_size, *hw).to(device)

    def do_epoch(loader, train):
        if args.refine:
            return run_epoch_refine(loader, depth_model, pose_enc, pose_dec, R, opt, opt0,
                                    ssim, backproj, project, device, args, hw, weights, train)
        return run_epoch(loader, depth_model, pose_enc, pose_dec, ssim, backproj, project,
                         opt, device, args, hw, train)

    import wandb
    wandb.init(project=os.getenv("WANDB_PROJECT", "rarp"),
               entity=os.getenv("WANDB_ENTITY", "nmgtue"), name=args.run_name,
               config=dict(task="depth", model="endodac-base-dvlora-r4", image_shape=hw,
                           intrinsics=k_norm, frame_stride=args.frame_stride,
                           bottom_crop_frac=args.bottom_crop_frac, epochs=args.epochs,
                           batch_size=args.batch_size, lr=args.lr, smoothness=args.smoothness,
                           automask=not args.no_automask, augment=not args.no_augment,
                           overlay_mask=not args.no_overlay_mask,
                           overlay_std_thresh=args.overlay_std_thresh, refine=args.refine,
                           init=str(args.init), data_root=str(root)))

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    # before-training qualitative panel
    wandb.log({"qual/panel": wandb.Image(
        qualitative_panel(depth_model, panel, hw, device, args.min_depth, args.max_depth),
        caption="epoch 0 (warm-start, before RARP fine-tune)"), "epoch": 0})

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        tr_logs = do_epoch(tr, train=True)
        va_logs = do_epoch(va, train=False)
        sched.step()
        lr = opt.param_groups[0]["lr"]
        print(f"epoch {ep}/{args.epochs}  train_photo={tr_logs['photo']:.4f}  "
              f"val_photo={va_logs['photo']:.4f}  pose_trans={tr_logs['pose_trans']:.4f}",
              flush=True)
        panel_img = qualitative_panel(depth_model, panel, hw, device, args.min_depth, args.max_depth)
        wandb.log({**{f"train/{k}": v for k, v in tr_logs.items()},
                   **{f"val/{k}": v for k, v in va_logs.items()}, "lr": lr, "epoch": ep,
                   "qual/panel": wandb.Image(panel_img, caption=f"epoch {ep} [rgb | depth]")})

        if va_logs["photo"] < best:                    # select by Validation photometric proxy
            best = va_logs["photo"]
            torch.save(depth_model.state_dict(), outdir / "best.pth")  # 389-key GUI state_dict
            wandb.run.summary["best_val_photo"] = best
            wandb.run.summary["best_epoch"] = ep

    # final report numbers on Test (proxy only -- no depth GT)
    te_ds = RARPTriplets(root / "Test", hw, k_norm, args.frame_stride, args.bottom_crop_frac, **ds_kw)
    te = DataLoader(te_ds, args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True)
    depth_model.load_state_dict(torch.load(outdir / "best.pth", map_location=device))
    te_logs = do_epoch(te, train=False)
    print(f"[test] photo={te_logs['photo']:.4f} smooth={te_logs['smooth']:.4f}", flush=True)
    wandb.run.summary.update({f"test/{k}": v for k, v in te_logs.items()})
    wandb.log({"qual/test_panel": wandb.Image(
        qualitative_panel(depth_model, [(te_ds[i][("color", 0)], te_ds[i]["valid"])
                          for i in np.linspace(0, len(te_ds) - 1, args.panel_size).astype(int)],
                          hw, device, args.min_depth, args.max_depth), caption="Test [rgb | depth]")})
    wandb.finish()
    print(f"[done] best val_photo={best:.4f} -> {outdir/'best.pth'}")


if __name__ == "__main__":
    main()
