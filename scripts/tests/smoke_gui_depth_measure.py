"""Headless smoke test for scripts/gui_depth_measure.py (Xvfb + --no-model).

Drives the real Tk window through the flow you actually use -- load, auto-crop, click-click
-> a measurement, scale/patch changes, calibrate, undo/clear, CSV + PNG export -- so the
class of bug you'd otherwise only find by opening the app on Windows fails here instead.
Modal dialogs are stubbed (a real one would block the mainloop forever, headless).

    xvfb-run -a python3.12 scripts/tests/smoke_gui_depth_measure.py
    xvfb-run -a python3.12 scripts/tests/smoke_gui_depth_measure.py /tmp/gui_smoke <ckpt.pth>

With a checkpoint it drives the REAL model instead of the synthetic ramp (any state_dict will
do -- scripts/tests/smoke_depth_backend.py can save a random one).
"""
import csv
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import gui_depth_measure as G                    # noqa: E402

TMP = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/gui_smoke")
CKPT = sys.argv[2] if len(sys.argv) > 2 else ""
TMP.mkdir(parents=True, exist_ok=True)
FAILS = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def make_frames(dirpath, n=3):
    return [make_frame(dirpath / f"frame_{i:04d}.png") for i in range(1, n + 1)]


def make_frame(path):
    """1920x1080 pillarboxed 5:4 'console frame': black bars + textured content."""
    a = np.zeros((1080, 1920, 3), np.uint8)
    yy, xx = np.mgrid[0:1080, 0:1344]
    a[:, 288:1632] = np.stack([(80 + 60 * np.sin(xx / 40)).astype(np.uint8),
                               (60 + 50 * np.cos(yy / 35)).astype(np.uint8),
                               np.full_like(xx, 70, np.uint8)], -1)
    Image.fromarray(a).save(path)
    return path


def stub_dialogs(csv_path, png_path, true_mm):
    """Replace every blocking dialog. Returns the list that records error/info popups."""
    import tkinter.filedialog as fd
    import tkinter.messagebox as mb
    import tkinter.simpledialog as sd
    popups = []
    for name in ("showerror", "showinfo", "showwarning"):
        setattr(mb, name, lambda title, msg=None, _n=name, **k: popups.append((_n, title, msg)))
    fd.asksaveasfilename = lambda **k: str(csv_path if k.get("defaultextension") == ".csv" else png_path)
    fd.askopenfilename = lambda **k: ""
    fd.askdirectory = lambda **k: ""
    sd.askfloat = lambda *a, **k: true_mm
    return popups


def drive(ctx):
    root, state, sess = ctx["root"], ctx["state"], ctx["sess"]
    csv_path, png_path = TMP / "m.csv", TMP / "m.png"
    popups = stub_dialogs(csv_path, png_path, true_mm=25.0)
    try:
        check(state["depth"] is not None, "depth predicted on load")
        check(f"build {G.BUILD}" in root.title(), f"title carries the build stamp ({G.BUILD})")
        panel = ctx["cache_lbl"].winfo_toplevel()
        check(panel is root, "control panel lives in the window")
        check(ctx["cache_lbl"].winfo_manager() == "pack", "cache label is laid out")
        check(not CKPT or ctx["provider"]().backend.model is not None,
              "real model loaded inside the GUI")
        check(state["crop"].size == (1344, 1080),
              f"auto-crop stripped the pillarbox -> {state['crop'].size}")
        check(abs(state["crop"].size[0] / state["crop"].size[1] - 1.25) < 0.01,
              "cropped view is 5:4, i.e. training framing")

        ctx["redraw"]()
        w, h = state["crop"].size

        def click(un, vn):
            z = state["zoom"]
            ctx["on_click"](types.SimpleNamespace(x=int(un * w * z), y=int(vn * h * z)))

        click(0.30, 0.50)
        check(state["pending"] is not None, "first click arms a pending point")
        click(0.70, 0.50)
        check(len(sess.items) == 1 and state["pending"] is None,
              "second click closes the measurement")

        m = sess.items[0]
        check(abs(m.p0[0] - 0.30) < 0.01 and abs(m.p1[0] - 0.70) < 0.01,
              "clicked pixels round-trip through the zoom transform")
        check(abs(m.mm - G.segment_length(m.p0, m.p1, m.z0, m.z1, sess.k_norm)) < 1e-6,
              f"reported {m.mm:.2f} mm equals the back-projected length")
        check(np.isfinite(m.mm) and m.mm > 0, "length is finite and positive")
        check(ctx["listbox"].size() == 1, "measurement appears in the list")

        ctx["on_motion"](types.SimpleNamespace(x=int(0.5 * w), y=int(0.5 * h)))

        ctx["vars"]["scale"].set("2.0"); ctx["apply_params"]()
        check(abs(sess.items[0].mm - 2 * m.mm) < 1e-6, "scale x2 doubles existing measurements")
        ctx["vars"]["scale"].set("1.0"); ctx["apply_params"]()
        check(abs(sess.items[0].mm - m.mm) < 1e-6, "scale back to 1 restores the original mm")

        ctx["vars"]["rad"].set("0"); ctx["apply_params"]()
        check(abs(sess.items[0].mm - m.mm) < 0.05 * m.mm, "patch radius 0 agrees within 5%")
        ctx["vars"]["rad"].set("2"); ctx["apply_params"]()

        n_pop = len(popups)
        ctx["vars"]["scale"].set("nonsense"); ctx["apply_params"]()
        check(len(popups) == n_pop + 1, "a non-numeric parameter shows an error instead of crashing")
        ctx["vars"]["scale"].set("1.0"); ctx["apply_params"]()

        ctx["listbox"].selection_clear(0, "end"); ctx["listbox"].selection_set(0)
        ctx["calibrate"]()
        check(abs(sess.items[0].mm - 25.0) < 1e-6,
              f"calibrate pins the selected segment to 25 mm (scale x{sess.scale:.4f})")
        ctx["vars"]["scale"].set("1.0"); ctx["apply_params"]()

        click(0.40, 0.30); click(0.60, 0.70)
        check(len(sess.items) == 2, "a second measurement can be added")
        m2 = sess.items[1]
        check(abs(m2.z0 - m2.z1) > 1e-3, "the two endpoints sample different depths")

        ctx["export_csv"]()
        rows = list(csv.DictReader(csv_path.open()))
        check(len(rows) == 2, f"CSV holds both measurements ({len(rows)} rows)")
        check(abs(float(rows[0]["length_mm"]) - sess.items[0].mm) < 0.01,
              "CSV length matches the UI")
        check(abs(float(rows[0]["x0"]) - m.p0[0] * w) < 1.0, "CSV coordinates are cropped-image px")

        ctx["save_png"]()
        check(png_path.exists() and Image.open(png_path).size == (w, h),
              "annotated PNG written at cropped-image size")

        ctx["vars"]["alpha"].set(1.0); ctx["redraw"]()
        ctx["vars"]["alpha"].set(0.0); ctx["redraw"]()
        ctx["zoom"](1.25); ctx["zoom"](1 / 1.25); ctx["zoom"](0.0)
        click(0.5, 0.5); click(0.9, 0.9)
        check(len(sess.items) == 3, "clicking still works after zoom changes")

        ctx["undo"]()
        check(len(sess.items) == 2, "undo removes the last measurement")
        ctx["clear"]()
        check(not sess.items and ctx["listbox"].size() == 0, "clear empties the session")

        # ---- cache: a frame is only ever predicted once per checkpoint + crop
        prov = ctx["provider"]()
        n0 = prov.computed
        first = state["depth"].copy()
        ctx["reload_image"](keep_marks=False)
        check(prov.computed == n0, "reloading the same frame does not run the model again")
        check(np.array_equal(state["depth"], first), "the cached map is bit-identical")
        prov.mem.clear()
        ctx["reload_image"](keep_marks=False)
        check(prov.computed == n0 and np.array_equal(state["depth"], first),
              "with the memory tier dropped, the map still comes off disk")
        cache_dir = prov.cache.dir
        check(cache_dir.exists() and (cache_dir / "meta.json").exists(),
              f"cache folder written with meta.json ({cache_dir.name})")
        npz = sorted(cache_dir.glob("*.npz"))
        check(len(npz) >= 1, f"depth map stored as .npz ({len(npz)})")
        with np.load(npz[0]) as z:
            check(np.array_equal(z["depth"], first) and z["depth"].dtype == np.float32,
                  "the stored .npz holds the same float32 map, readable by other scripts")

        ctx["vars"]["side"].set("0.30")                      # a different crop is a different map
        ctx["reload_image"](keep_marks=False)
        check(prov.computed == n0 + 1, "changing the crop recomputes rather than reusing")
        ctx["set_auto_crop"]()
        check(prov.computed == n0 + 1, "going back to the auto crop hits the cache again")

        n1 = prov.computed
        ctx["precompute_all"]()
        check(prov.computed == n1 + 2, f"precompute filled the other 2 frames (+{prov.computed - n1})")
        ctx["precompute_all"]()
        check(prov.computed == n1 + 2, "a second precompute run has nothing to do")
        check(ctx["filebox"].get(0).startswith("*"), "cached frames are marked in the file list")
        ctx["select_index"](2)
        check(prov.computed == n1 + 2, "stepping to a precomputed frame runs no model")

        # ---- a different checkpoint means a different folder, and back again is free
        if CKPT:
            n_maps = prov.cache.count()
            ok = ctx["set_checkpoint"](str(CKPT))             # reload -> same fingerprint, same dir
            prov2 = ctx["provider"]()
            check(ok and prov2.cache.dir == cache_dir,
                  "reselecting a checkpoint returns to its own cache folder")
            check(prov2.cache.count() == n_maps and prov2.computed == 0,
                  f"and its {n_maps} maps are picked straight back up, nothing recomputed")

        ctx["vars"]["side"].set("0.0"); ctx["vars"]["top"].set("0.0"); ctx["vars"]["bot"].set("0.0")
        ctx["reload_image"](keep_marks=False)
        check(state["crop"].size == (1920, 1080), "manual crop fracs are honoured (full frame)")
        ctx["set_auto_crop"]()
        check(state["crop"].size == (1344, 1080), "Auto bars restores the 5:4 content")
    except Exception as exc:                              # noqa: BLE001
        import traceback
        traceback.print_exc()
        FAILS.append(f"exception: {exc}")
    finally:
        root.destroy()


def main():
    frames = make_frames(TMP)
    img = frames[0]
    args = types.SimpleNamespace(
        ckpt=CKPT, data_dir=str(TMP), image=str(img), image_shape=list(G.DEFAULT_SHAPE),
        cache_dir=str(TMP / "cache"), no_cache=False,
        min_depth=G.DEFAULT_MIN_DEPTH, max_depth=G.DEFAULT_MAX_DEPTH,
        intrinsics=list(G.DEFAULT_K_NORM), scale=1.0, patch_radius=2, device="cpu",
        no_model=not CKPT, self_test=False, version=False)
    G.run_gui(args, on_ready=drive)
    print("FAILURES:", FAILS if FAILS else "none")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
