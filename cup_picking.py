"""Multi-view RGB-D cup picking: final inference code (experiment 8a).

Predictor(checkpoint, device).predict(observations, request) -> grasp prediction
  observations[view] = {"rgb": uint8 [H, W, 3], "geometry": {depth_m, depth_valid, K, camera_to_world}}
  for view in ("sw", "se", "ne", "nw"); request = {"text", "color", "pattern"}.

This file contains only our code. The course-provided definitions it relies on (BaselineCNN,
image_to_tensor, point_from_depth, decode_heatmap, VIEWS/COLORS/PATTERNS) are NOT redistributed here;
they are loaded at import time from the course's baseline.ipynb (cells tagged "cafe.inference").
Set CAFE_COURSE_DIR to the course project folder (the one containing baseline.ipynb).
"""
import json
import os
from pathlib import Path


def _load_course_definitions():
    notebook = Path(os.environ.get("CAFE_COURSE_DIR", ".")) / "baseline.ipynb"
    if not notebook.is_file():
        raise FileNotFoundError(f"{notebook} not found: set CAFE_COURSE_DIR to the course project folder")
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    source = "\n\n".join("".join(c["source"]) for c in cells
                          if c["cell_type"] == "code" and "cafe.inference" in c.get("metadata", {}).get("tags", []))
    namespace = {"__name__": "course_baseline"}
    exec(compile(source, str(notebook), "exec"), namespace)
    return {k: v for k, v in namespace.items() if not k.startswith("__") and k != "Predictor"}


globals().update(_load_course_definitions())   # course-provided: see module docstring

import math                                     # noqa: E402
import numpy as np                              # noqa: E402
from PIL import Image                           # noqa: E402
import torch                                    # noqa: E402
from torch import nn                            # noqa: E402
from torch.nn import functional as F            # noqa: E402


# Improved pipeline, part 1: depth cup candidates and a self-contained DINOv2 ViT-S/14.
# Inference inputs only: RGB, depth, depth validity, K, camera_to_world, and the request.
# No labels, no downloads: DINOv2 weights are read from the improved checkpoint.

CANDIDATE_TABLE_TOL = 0.004          # RANSAC inlier distance to the table plane [m]
CANDIDATE_MAX_TILT_DEG = 15          # the table plane is roughly horizontal (world z is up)
CANDIDATE_HEIGHT = (0.006, 0.20)     # points kept above the table [m]
CANDIDATE_VOXEL = 0.008              # Euclidean clustering tolerance (26-neighbour voxels) [m]
CANDIDATE_MIN_POINTS = 200
CANDIDATE_MAX_XY = 0.20              # wider clusters are walls or junk [m]
CANDIDATE_MIN_TOP = 0.03             # a cup rises at least 3 cm above the table [m]
CROP_SIZE, CROP_MARGIN, CROP_GRAY, CROP_MIN_PIXELS = 224, 0.10, 128, 150
DINO_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
DINO_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def view_world_points(geometry):
    """All pixels with valid depth -> (rows, cols, world points [N, 3])."""
    depth = geometry["depth_m"]
    valid = geometry["depth_valid"] & np.isfinite(depth) & (depth > 0)
    rows, cols = np.nonzero(valid)
    z = depth[rows, cols].astype(np.float64)
    rays = np.stack([cols, rows, np.ones_like(cols)], 1).astype(np.float64) @ np.linalg.inv(geometry["K"]).T
    transform = np.asarray(geometry["camera_to_world"], dtype=np.float64)
    return rows, cols, (rays * z[:, None]) @ transform[:3, :3].T + transform[:3, 3]


def fit_table_plane(points, rng, iterations=300):
    """RANSAC over near-horizontal planes; the table is the one with most inliers."""
    sample = points[rng.choice(len(points), min(30000, len(points)), replace=False)]
    min_up = math.cos(math.radians(CANDIDATE_MAX_TILT_DEG))
    best_count, best = -1, None
    for _ in range(iterations):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal = normal / length
        if normal[2] < 0:
            normal = -normal
        if normal[2] < min_up:
            continue
        count = int((np.abs(sample @ normal - normal @ a) < CANDIDATE_TABLE_TOL).sum())
        if count > best_count:
            best_count, best = count, (normal, -normal @ a)
    if best is None:
        raise ValueError("No horizontal table plane found")
    normal, offset = best
    inliers = sample[np.abs(sample @ normal + offset) < CANDIDATE_TABLE_TOL]
    centre = inliers.mean(0)
    normal = np.linalg.svd(inliers - centre, full_matrices=False)[2][-1]
    normal = normal if normal[2] > 0 else -normal
    return normal, -normal @ centre, inliers


def euclidean_clusters(points):
    """Connected components of occupied voxels (26-neighbourhood); one label per point."""
    keys = np.floor(points / CANDIDATE_VOXEL).astype(np.int64)
    keys -= keys.min(0)
    size = int(keys.max()) + 3
    codes = (keys[:, 0] * size + keys[:, 1]) * size + keys[:, 2]
    unique, inverse = np.unique(codes, return_inverse=True)
    voxels = np.stack([unique // (size * size), (unique // size) % size, unique % size], 1)
    edges = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if (dx, dy, dz) <= (0, 0, 0):
                    continue
                neighbour = voxels + (dx, dy, dz)
                code = (neighbour[:, 0] * size + neighbour[:, 1]) * size + neighbour[:, 2]
                index = np.clip(np.searchsorted(unique, code), 0, len(unique) - 1)
                hit = unique[index] == code
                edges.append(np.stack([np.nonzero(hit)[0], index[hit]], 1))
    edges = np.concatenate(edges)
    label = np.arange(len(unique))
    while True:
        updated = label.copy()
        np.minimum.at(updated, edges[:, 0], label[edges[:, 1]])
        np.minimum.at(updated, edges[:, 1], label[edges[:, 0]])
        updated = updated[updated]
        if np.array_equal(updated, label):
            break
        label = updated
    return np.unique(label, return_inverse=True)[1][inverse]


def propose_candidates(observations, views, seed=17):
    """Cup candidates from depth: table removal + 3D Euclidean clustering. A boolean mask per view."""
    rng = np.random.default_rng(seed)
    per_view = {view: view_world_points(observations[view]["geometry"]) for view in views}
    normal, offset, table = fit_table_plane(np.concatenate([p for _, _, p in per_view.values()]), rng)
    low = np.percentile(table[:, :2], 0.5, axis=0) - 0.01
    high = np.percentile(table[:, :2], 99.5, axis=0) + 0.01
    kept, where = [], []
    for view_id, view in enumerate(views):
        rows, cols, points = per_view[view]
        height = points @ normal + offset
        keep = ((height > CANDIDATE_HEIGHT[0]) & (height < CANDIDATE_HEIGHT[1])
                & np.all((points[:, :2] >= low) & (points[:, :2] <= high), 1))
        kept.append(points[keep])
        where.append(np.stack([np.full(int(keep.sum()), view_id), rows[keep], cols[keep]], 1))
    points, where = np.concatenate(kept), np.concatenate(where)
    if len(points) == 0:
        return []
    labels = euclidean_clusters(points)
    candidates = []
    for label in np.unique(labels):
        member = labels == label
        cluster = points[member]
        if len(cluster) < CANDIDATE_MIN_POINTS:
            continue
        if np.ptp(cluster[:, 0]) > CANDIDATE_MAX_XY or np.ptp(cluster[:, 1]) > CANDIDATE_MAX_XY:
            continue
        if (cluster @ normal + offset).max() < CANDIDATE_MIN_TOP:
            continue
        masks = {}
        for view_id, view in enumerate(views):
            pixels = where[member & (where[:, 0] == view_id)]
            mask = np.zeros(observations[view]["geometry"]["depth_m"].shape, bool)
            mask[pixels[:, 1], pixels[:, 2]] = True
            masks[view] = mask
        candidates.append({"centre": cluster.mean(0), "masks": masks})
    return candidates


def masked_crop(rgb, mask):
    """Square crop around a mask, gray outside the mask, resized for DINOv2. None if too small."""
    if int(mask.sum()) < CROP_MIN_PIXELS:
        return None
    image = rgb.copy()
    image[~mask] = CROP_GRAY
    ys, xs = np.nonzero(mask)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    side = int(math.ceil(max(ys.max() - ys.min(), xs.max() - xs.min()) * (1 + 2 * CROP_MARGIN))) + 1
    padded = np.pad(image, ((side, side), (side, side), (0, 0)), constant_values=CROP_GRAY)
    top, left = int(round(cy - side / 2)) + side, int(round(cx - side / 2)) + side
    crop = padded[top:top + side, left:left + side]
    return np.array(Image.fromarray(crop).resize((CROP_SIZE, CROP_SIZE), Image.BICUBIC))


class DinoBlock(nn.Module):
    """One DINOv2 transformer block (LayerScale, plain attention). Parameter names match DINOv2."""

    def __init__(self, dim=384, heads=6):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(dim, dim * 3)
        self.attn.proj = nn.Linear(dim, dim)
        self.ls1 = nn.Module()
        self.ls1.gamma = nn.Parameter(torch.ones(dim))
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(dim, dim * 4)
        self.mlp.fc2 = nn.Linear(dim * 4, dim)
        self.ls2 = nn.Module()
        self.ls2.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.attn.qkv(self.norm1(x)).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        attention = ((qkv[0] * (c // self.heads) ** -0.5) @ qkv[1].transpose(-2, -1)).softmax(-1)
        x = x + self.ls1.gamma * self.attn.proj((attention @ qkv[2]).transpose(1, 2).reshape(b, n, c))
        return x + self.ls2.gamma * self.mlp.fc2(F.gelu(self.mlp.fc1(self.norm2(x))))


class DinoViTS14(nn.Module):
    """DINOv2 ViT-S/14 feature extractor. Output: CLS token + mean patch token [B, 768]."""

    def __init__(self, dim=384, depth=12, heads=6, patch=14, grid=37):
        super().__init__()
        self.patch = patch
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, dim, patch, patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, grid * grid + 1, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, dim))
        self.blocks = nn.ModuleList([DinoBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def positions(self, w, h):
        pos = self.pos_embed.float()
        m = int(math.sqrt(pos.shape[1] - 1))
        w0, h0 = w // self.patch, h // self.patch
        if w0 * h0 == m * m and w == h:
            return self.pos_embed
        grid = pos[:, 1:].reshape(1, m, m, -1).permute(0, 3, 1, 2)
        grid = F.interpolate(grid, mode="bicubic", antialias=False,
                             scale_factor=((w0 + 0.1) / m, (h0 + 0.1) / m))
        return torch.cat([pos[:, :1], grid.permute(0, 2, 3, 1).reshape(1, -1, pos.shape[-1])], 1)

    def forward(self, x):
        _, _, w, h = x.shape
        tokens = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
        tokens = torch.cat([self.cls_token.expand(len(tokens), -1, -1), tokens], 1) + self.positions(w, h)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return torch.cat([tokens[:, 0], tokens[:, 1:].mean(1)], 1)


# Public robot workspace (robot/backend/open_lift.py), shrunk by a small margin.
WORKSPACE_LOW = np.array([0.07, -0.63, 0.02])
WORKSPACE_HIGH = np.array([0.83, 0.48, 0.20])


def inside_workspace(point):
    point = np.asarray(point, dtype=float)
    return bool(np.all(point >= WORKSPACE_LOW) and np.all(point <= WORKSPACE_HIGH))


# Geometric handle grasp (8a). Margin chosen on TRAIN in 7a (smallest margin with 100 % handle precision).
HANDLE_MARGIN_M = 0.008
MIN_HANDLE_PX = 30
OUTER_QUANTILE = 0.9


def fit_body_circle(xy, seed=17, iterations=300, tol=0.003):
    """RANSAC circle in the table plane; the body wall gives the most inliers -> (centre, radius)."""
    if len(xy) < 30:
        return None
    rng = np.random.default_rng(seed)
    if len(xy) > 3000:
        xy = xy[rng.choice(len(xy), 3000, replace=False)]
    best, best_n = None, 0
    for _ in range(iterations):
        a, b, c = xy[rng.choice(len(xy), 3, replace=False)]
        ab, ac = b - a, c - a
        det = 2 * (ab[0] * ac[1] - ab[1] * ac[0])
        if abs(det) < 1e-9:
            continue
        qb, qc = ab @ ab, ac @ ac
        centre = a + np.array([qb * ac[1] - qc * ab[1], qc * ab[0] - qb * ac[0]]) / det
        radius = float(np.linalg.norm(a - centre))
        if not 0.025 <= radius <= 0.075:
            continue
        n = int((np.abs(np.linalg.norm(xy - centre, axis=1) - radius) < tol).sum())
        if n > best_n:
            best, best_n = (centre, radius), n
    return best


class Predictor:
    """Final model (8a): which cup (depth candidates + DINOv2 colour/pattern) and where (outer end of the
    depth-detected handle; 1b heatmap only as fallback).

    1. Cup candidates from depth: table-plane removal + 3D Euclidean clustering
       (views from the checkpoint config; default: all four views fused in world coordinates).
    2. Each candidate is cropped from every view (SW/SE/NE/NW masks) and scored by frozen DINOv2
       features with linear colour and pattern heads. Pattern: max over views of P(pattern) for a
       patterned request, min over views for "plain" (one view showing a pattern is enough);
       colour: mean over views. Colour filter: keep P(colour) >= color_min, then pick the
       highest P(colour) * P(pattern).
    3. The 1b heatmap (trained on four views) is restricted to the chosen cup's mask in SW (or,
       if the cup is barely visible in SW, in the view where it is largest); its maximum is
       back-projected with that view's depth.
    Fallback (no candidate, no candidate of the requested colour, unreadable crops, invalid depth
    or workspace): the 1b four-view
    prediction ranked by observability.

    Grasp: a RANSAC circle on the chosen cup's 3D points (all views) gives the body; mask pixels more
    than HANDLE_MARGIN_M outside it are the handle; the grasp pixel is the median of the outer 10 % of
    handle pixels, snapped to the nearest handle pixel with valid depth inside the workspace. If no
    handle is found, the heatmap grasp above (same cup, other views) is used.

    The checkpoint bundles the 1b grasp network, DINOv2 ViT-S/14 weights, the two heads and the
    configuration, so inference needs no downloads and no labels.
    """

    def __init__(self, checkpoint=None, device="cpu"):
        if checkpoint is None:
            raise ValueError("A checkpoint path is required.")
        bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.device = device
        self.model = BaselineCNN().to(device).eval()
        self.model.load_state_dict(bundle["state_dict"])
        self.dino = DinoViTS14().to(device).eval()
        self.dino.load_state_dict(bundle["dinov2_state_dict"])
        self.heads = {name: {key: value.float().to(device) for key, value in head.items()}
                      for name, head in bundle["heads"].items()}
        config = bundle.get("config", {})
        self.candidate_views = tuple(config.get("candidate_views", VIEWS))
        self.color_min = float(config.get("color_min", 0.5))

    # ---------------------------------------------------------------- heads
    def _head(self, name, features):
        head = self.heads[name]
        logits = ((features - head["mean"]) / head["std"]) @ head["weight"].T + head["bias"]
        return logits.softmax(1).cpu().numpy()

    def _score(self, candidates, observations, request):
        """Per-view crops of every candidate -> fused P(colour), P(pattern) for the request."""
        views = [view for view in VIEWS if view in observations]
        jobs, crops = [], []
        for index, candidate in enumerate(candidates):
            for view in views:
                mask = candidate["masks"].get(view)
                crop = None if mask is None else masked_crop(observations[view]["rgb"], mask)
                if crop is not None:
                    jobs.append(index)
                    crops.append(crop)
        p_color = np.zeros(len(candidates))
        p_pattern = np.zeros(len(candidates))
        readable = sorted(set(jobs))
        if not crops:
            return readable, p_color, p_pattern
        color_id, pattern_id = COLORS.index(request["color"]), PATTERNS.index(request["pattern"])
        colors, patterns = [], []
        for start in range(0, len(crops), 32):
            batch = torch.from_numpy(np.stack(crops[start:start + 32])).permute(0, 3, 1, 2).float() / 255
            features = self.dino(((batch - DINO_MEAN) / DINO_STD).to(self.device))
            colors.append(self._head("color", features)[:, color_id])
            patterns.append(self._head("pattern", features)[:, pattern_id])
        colors, patterns, jobs = np.concatenate(colors), np.concatenate(patterns), np.array(jobs)
        plain = request["pattern"] == "plain"
        for index in readable:
            mine = jobs == index
            p_color[index] = colors[mine].mean()
            # A pattern seen in any view is evidence for that pattern; "plain" needs every view to agree.
            p_pattern[index] = patterns[mine].min() if plain else patterns[mine].max()
        return readable, p_color, p_pattern

    def _grasp_view(self, candidate, observations):
        """SW if the cup is visible enough there (as in 1c), otherwise the view where it is largest."""
        masks = candidate["masks"]
        if int(masks["sw"].sum()) >= CROP_MIN_PIXELS:
            return "sw"
        views = [view for view in VIEWS if view in observations and view in masks]
        return max(views, key=lambda view: int(masks[view].sum()))

    # ---------------------------------------------------------------- grasp
    def _heatmap(self, rgb, request):
        image = image_to_tensor(rgb).unsqueeze(0).to(self.device)
        color = torch.tensor([COLORS.index(request["color"])], device=self.device)
        pattern = torch.tensor([PATTERNS.index(request["pattern"])], device=self.device)
        return self.model(image, color, pattern)

    @staticmethod
    def _world_image(geometry):
        rows, cols, points = view_world_points(geometry)
        image = np.full(geometry["depth_m"].shape + (3,), np.nan)
        image[rows, cols] = points
        return image

    def _handle_grasp(self, candidate, observations):
        """Outer end of the handle found from depth geometry -> (view, pixel, world) or None."""
        views = [v for v in VIEWS if v in observations and v in candidate["masks"]]
        world = {v: self._world_image(observations[v]["geometry"]) for v in views}
        points = np.concatenate([world[v][candidate["masks"][v]] for v in views])
        points = points[np.isfinite(points).all(1) & (points[:, 2] > 0.02)]
        fit = fit_body_circle(points[:, :2])
        if fit is None:
            return None
        centre, radius = fit
        handles = {}
        for v in views:
            radial = np.linalg.norm(world[v][..., :2] - centre, axis=-1)
            handles[v] = (candidate["masks"][v] & np.nan_to_num(radial > radius + HANDLE_MARGIN_M), radial)
        counts = {v: int(h.sum()) for v, (h, _) in handles.items()}
        view = "sw" if counts.get("sw", 0) >= MIN_HANDLE_PX else max(counts, key=counts.get)
        if counts[view] < MIN_HANDLE_PX:
            return None
        handle, radial = handles[view]
        ys, xs = np.nonzero(handle)
        r = radial[ys, xs]
        outer = r >= np.quantile(r, OUTER_QUANTILE)
        target = np.array([np.median(xs[outer]), np.median(ys[outer])])
        geometry = observations[view]["geometry"]
        for j in np.argsort((xs - target[0]) ** 2 + (ys - target[1]) ** 2)[:200]:
            pixel = np.array([xs[j], ys[j]], dtype=float)
            try:
                point, _ = point_from_depth(pixel, geometry)
            except ValueError:
                continue
            if inside_workspace(point):
                return view, pixel, point
        return None

    def _grasp_same_cup(self, candidate, observations, request):
        """Geometric handle grasp first; otherwise the 1e heatmap grasp on the same cup."""
        found = self._handle_grasp(candidate, observations)
        if found is not None:
            view, pixel, point = found
            return view, (pixel, point), "geometric handle (outer end)"
        return self._grasp_heatmap_same_cup(candidate, observations, request)

    def _snap_to_mask(self, geometry, mask, pixel):
        """Nearest mask pixel with valid depth whose back-projection is inside the workspace."""
        depth = geometry["depth_m"]
        usable = mask & geometry["depth_valid"] & np.isfinite(depth) & (depth > 0)
        ys, xs = np.nonzero(usable)
        for nearest in np.argsort((xs - pixel[0]) ** 2 + (ys - pixel[1]) ** 2)[:200]:
            candidate = np.array([xs[nearest], ys[nearest]], dtype=float)
            world, _ = point_from_depth(candidate, geometry)
            if inside_workspace(world):
                return candidate, world
        return None

    def _grasp_heatmap_same_cup(self, candidate, observations, request):
        """1d grasp; if invalid, snap to the mask in that view, then try the cup's other views."""
        first = self._grasp_view(candidate, observations)
        grasp = self._grasp_in_mask(observations[first], request, candidate["masks"][first])
        if grasp is not None:
            return first, grasp, "heatmap"
        others = sorted((v for v in VIEWS if v in observations and v != first and v in candidate["masks"]),
                        key=lambda v: -int(candidate["masks"][v].sum()))
        for step, view in enumerate([first] + others):
            mask = candidate["masks"][view]
            if int(mask.sum()) < CROP_MIN_PIXELS:
                continue
            if step == 0:
                logits = self._heatmap(observations[view]["rgb"], request)["heatmap_logits"]
                cells = F.adaptive_max_pool2d(torch.from_numpy(mask).float()[None, None], logits.shape[-2:])
                normalized = decode_heatmap(logits.masked_fill(~cells[0].bool().to(logits.device), -1e9))[0].cpu().numpy()
                rgb = observations[view]["rgb"]
                pixel = normalized * np.array([rgb.shape[1], rgb.shape[0]]) - 0.5
                grasp = self._snap_to_mask(observations[view]["geometry"], mask, pixel)
                if grasp is not None:
                    return view, grasp, "snapped to mask"
            else:
                grasp = self._grasp_in_mask(observations[view], request, mask)
                if grasp is not None:
                    return view, grasp, "other view"
        return None

    def _grasp_in_mask(self, observation, request, mask):
        rgb, geometry = observation["rgb"], observation["geometry"]
        logits = self._heatmap(rgb, request)["heatmap_logits"]
        cells = F.adaptive_max_pool2d(torch.from_numpy(mask).float()[None, None], logits.shape[-2:])
        allowed = cells[0].bool().to(logits.device)
        normalized = decode_heatmap(logits.masked_fill(~allowed, -1e9))[0].cpu().numpy()
        pixel = normalized * np.array([rgb.shape[1], rgb.shape[0]]) - 0.5
        try:
            world, _ = point_from_depth(pixel, geometry)
        except ValueError:
            depth = geometry["depth_m"]
            usable = mask & geometry["depth_valid"] & np.isfinite(depth) & (depth > 0)
            ys, xs = np.nonzero(usable)
            if len(ys) == 0:
                return None
            nearest = int(np.argmin((xs - pixel[0]) ** 2 + (ys - pixel[1]) ** 2))
            pixel = np.array([xs[nearest], ys[nearest]], dtype=float)
            world, _ = point_from_depth(pixel, geometry)
        if not inside_workspace(world):
            return None
        return pixel, world

    # ---------------------------------------------------------------- fallback (1b four views)
    def _four_view(self, observations, request, note):
        views = [view for view in VIEWS if view in observations]
        image_batch = torch.stack([image_to_tensor(observations[v]["rgb"]) for v in views]).to(self.device)
        color = torch.full((len(views),), COLORS.index(request["color"]), device=self.device)
        pattern = torch.full((len(views),), PATTERNS.index(request["pattern"]), device=self.device)
        out = self.model(image_batch, color, pattern)
        normalized = decode_heatmap(out["heatmap_logits"]).cpu().numpy()
        probabilities = out["observable_logit"].sigmoid().cpu().numpy()
        for index in np.argsort(-probabilities):
            view = views[index]
            rgb = observations[view]["rgb"]
            pixel = normalized[index] * np.array([rgb.shape[1], rgb.shape[0]]) - 0.5
            try:
                world, _ = point_from_depth(pixel, observations[view]["geometry"])
            except ValueError:
                continue
            if inside_workspace(world):
                return {"status": "PICK", "selected_view": view, "pixel_uv": pixel.tolist(),
                        "world_xyz_m": np.asarray(world).tolist(), "pipeline": "four_view_fallback",
                        "fallback_reason": note}
        return {"status": "NO_PICK", "reason": "No usable view. " + note, "pipeline": "four_view_fallback"}

    @torch.inference_mode()
    def predict(self, observations, request):
        """observations[view] = {rgb, geometry}; request = {text, color, pattern} -> grasp prediction."""
        if "sw" not in observations or not all(view in observations for view in self.candidate_views):
            return self._four_view(observations, request, "candidate views missing")
        try:
            candidates = propose_candidates(observations, self.candidate_views)
        except ValueError as error:
            return self._four_view(observations, request, f"candidates failed: {error}")
        if not candidates:
            return self._four_view(observations, request, "no cup candidates")

        readable, p_color, p_pattern = self._score(candidates, observations, request)
        if not readable:
            return self._four_view(observations, request, "no readable candidate")
        pool = [i for i in readable if p_color[i] >= self.color_min]   # colour filter
        if not pool:
            # No candidate has the requested colour (e.g. the target merged with another cup):
            # choosing among other colours is always wrong, so use the 1b four-view prediction.
            return self._four_view(observations, request, "no candidate passed the colour filter")
        ranked = sorted(pool, key=lambda i: p_color[i] * p_pattern[i], reverse=True)

        # Keep the chosen cup: a second-ranked candidate is a different cup (1c/1d: 1 success in 17).
        index = ranked[0]
        found = self._grasp_same_cup(candidates[index], observations, request)
        if found is not None:
            view, (pixel, world), how = found
            return {"status": "PICK", "selected_view": view, "pixel_uv": pixel.tolist(),
                    "world_xyz_m": np.asarray(world).tolist(), "pipeline": "candidates",
                    "candidate_count": len(candidates), "candidate_rank": 0, "grasp_source": how,
                    "color_probability": float(p_color[index]),
                    "pattern_probability": float(p_pattern[index])}
        return self._four_view(observations, request, "grasp point invalid for the chosen cup")
