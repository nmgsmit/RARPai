#!/usr/bin/env python
"""Reshape cut_cue_clips output into the layout finetune_depth expects.

cut_cue_clips --frames writes   <src>/<video>/cues/clip_NNN/images/frame_*.jpg
finetune_depth globs            <root>/<split>/*/*/clip_*/images   (finetune_depth.py:166)
                                i.e. <root>/<split>/rarp/<video>/clip_NNN/images

So the "cues" level has to go and the videos have to be split. The split is COPIED
from an existing dataset (--split-ref) rather than invented: comparing a new training
set against an old run is only meaningful if Validation and Test hold the same videos.
A video the reference does not know about is an error, not a silent Train.

    python scripts/assemble_depth_clips.py --src ../data/depth_clips_staging \
        --dst ../data/depth_clips \
        --split-ref ../data/UMCdissectionvid/UMCdissectionimg
"""
import argparse
import collections
import os
import shutil
import sys

SPLITS = ("Train", "Validation", "Test")


def split_map(ref):
    """{video stem: split} from an existing <ref>/<split>/rarp/<video> tree."""
    m = {}
    for sp in SPLITS:
        d = os.path.join(ref, sp, "rarp")
        if not os.path.isdir(d):
            continue
        for v in os.listdir(d):
            m[v] = sp
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="cut_cue_clips --out directory")
    p.add_argument("--dst", required=True, help="dataset root to build")
    p.add_argument("--split-ref", required=True, help="existing dataset defining the split")
    p.add_argument("--category", default="rarp", help="the level between split and video")
    p.add_argument("--copy", action="store_true", help="copy instead of move")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    smap = split_map(args.split_ref)
    if not smap:
        raise SystemExit(f"no split found under {args.split_ref}")
    print(f"split reference: {len(smap)} videos "
          f"({collections.Counter(smap.values()).most_common()})")

    moved = collections.Counter()
    frames = collections.Counter()
    unknown = []
    for video in sorted(os.listdir(args.src)):
        cues = os.path.join(args.src, video, "cues")
        if not os.path.isdir(cues):
            continue                       # video had no cue -> nothing was written
        sp = smap.get(video)
        if sp is None:
            unknown.append(video)
            continue
        for clip in sorted(os.listdir(cues)):
            srcd = os.path.join(cues, clip)
            imgs = os.path.join(srcd, "images")
            if not os.path.isdir(imgs):
                continue
            n = len([f for f in os.listdir(imgs) if f.endswith(".jpg")])
            dstd = os.path.join(args.dst, sp, args.category, video, clip)
            if not args.dry_run:
                os.makedirs(os.path.dirname(dstd), exist_ok=True)
                if os.path.exists(dstd):
                    shutil.rmtree(dstd)
                (shutil.copytree if args.copy else shutil.move)(srcd, dstd)
            moved[sp] += 1
            frames[sp] += n

    for sp in SPLITS:
        print(f"{sp:11s} {moved[sp]:5d} clips  {frames[sp]:7d} frames")
    print(f"{'TOTAL':11s} {sum(moved.values()):5d} clips  {sum(frames.values()):7d} frames")
    if unknown:
        # loud: these frames would silently never train
        print(f"\nWARNING: {len(unknown)} video(s) not in the split reference, SKIPPED:",
              file=sys.stderr)
        for v in unknown:
            print(f"  {v}", file=sys.stderr)


if __name__ == "__main__":
    main()
