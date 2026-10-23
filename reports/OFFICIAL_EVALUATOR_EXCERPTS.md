# Official evaluator: source-review excerpts

Public reference: https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook

Quoted solely for scientific evaluation review, not executed or incorporated into model code.

SHA256: `ae7c9445b4eee2b77b216df5c9772fdf92fda15611e98abd7ef5f02a4e380801`

```python
# Helper functions

def get_overlap_matrices(
        gt_layers: torch.Tensor,
        pred_layers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes pairwise IoU and Dice scores between ground-truth (GT) and predicted mask layers.
    This is called for all segmentations corresponding to one image.

    Args:
        gt_layers: Ground-truth layers, shape (n_gt, H, W), values in {0, 1}.
        pred_layers: Prediction layers, shape (n_pred, H, W), values in {0, 1}.

    Returns:
        A tuple ``(iou_matrix, dice_matrix)`` of tensors, each of shape
        (n_gt, n_pred) with values in [0, 1], where:

            iou_matrix[i, j]  = IoU(GT layer i, predicted layer j)
            dice_matrix[i, j] = Dice(GT layer i, predicted layer j)

        - **Rows (i, size n_gt)**: ground-truth filaments
        - **Columns (j, size n_pred)**: predicted filaments

        Example for n_gt=2 GT filaments and n_pred=3 predictions:

            iou_matrix = [[IoU(gt_0, pred_0), IoU(gt_0, pred_1), IoU(gt_0, pred_2)],
                          [IoU(gt_1, pred_0), IoU(gt_1, pred_1), IoU(gt_1, pred_2)]]
    """
    n_gt, height, width = gt_layers.shape
    n_pred = pred_layers.shape[0]

    gt_flat = gt_layers.reshape(n_gt, height * width)
    pred_flat = pred_layers.reshape(n_pred, height * width)

    intersection = torch.matmul(gt_flat, pred_flat.t())

    gt_areas = gt_flat.sum(dim=1).view(-1, 1)
    pred_areas = pred_flat.sum(dim=1).view(1, -1)

    union = gt_areas + pred_areas - intersection

    iou_matrix = torch.where(
        union == 0,
        torch.tensor(0., device=gt_layers.device),
        intersection / union,
    )

    dice_matrix = torch.where(
        union == 0,
        torch.tensor(0., device=gt_layers.device),
        2 * intersection / (gt_areas + pred_areas),
    )

    return iou_matrix, dice_matrix

def fp_count_hit(hit_matrix: torch.Tensor) -> int:
    """
    Counts false-positive predictions for one image from a matching matrix where
    the value is 1 if the predicted filament overlaps with the GT filament and 0 otherwise.

    A predicted filament is a false positive if it does not hit any GT filament:
    the sum of its column is zero.

    Example (rows = GT, columns = predictions):

        hit_matrix = [[0, 0, 1],
                      [0, 0, 0],
                      [0, 0, 1]]

        # column sums: [0, 0, 2]  ->  columns 0 and 1 are FPs
        fp_count_hit(hit_matrix) -> 2

    Args:
        hit_matrix: Boolean tensor of shape (n_gt, n_pred), match indicators per pair.

    Returns:
        Scalar integer tensor: number of predictions with no qualifying GT overlap.
    """
    gt_matches_per_pred = hit_matrix.sum(dim=0)
    return (gt_matches_per_pred == 0).sum().item()

def fn_count_hit(hit_matrix: torch.Tensor) -> int:
    """
    Counts false-negative annotations for one image from a matching matrix where
    the value is 1 if the predicted filament overlaps with the GT filament and 0 otherwise.

    A GT filament is a false negative if it does not hit any predicted filament: the sum of
    its row is zero.

    Example (rows = GT, columns = predictions):

        hit_matrix = [[0, 0, 1],
                      [0, 0, 0],
                      [0, 0, 1]]

        # row sums: [1, 0, 1]  ->  row 1 is FN
        fn_count_hit(hit_matrix) -> 1

    Args:
        hit_matrix: Boolean tensor of shape (n_gt, n_pred), match indicators per pair.

    Returns:
        Scalar integer tensor: number of annotations with no qualifying prediction overlap.
    """
    pred_matches_per_gt = hit_matrix.sum(dim=1)
    return (pred_matches_per_gt == 0).sum().item()

def rles_to_layers(rles: list[str], height: int = 2048, width: int = 2048) -> np.ndarray:
    """
    Decode a list of compressed COCO RLE strings into an (n_masks, H, W) binary mask stack.

    Args:
        rles: Compressed COCO RLE strings, one per filament.
        height: Mask height in pixels.
        width: Mask width in pixels.

    Returns:
        float32 array of shape (n_masks, H, W) with values in {0, 1}, where
        n_masks is ``len(rles)``. Returns an empty (0, H, W) array if ``rles``
        is empty.
    """
    if not rles:
        return np.zeros((0, height, width), dtype=np.float32)
    rle_dicts = [{"size": [height, width], "counts": rle} for rle in rles]
    # decode -> (H, W, n_masks), transpose to (n_masks, H, W)
    masks = mask_util.decode(rle_dicts)
    return masks.transpose(2, 0, 1).astype(np.float32)
    
def process_entry(
        annotator_image: str,
        gt_annotator_image_ids: pd.Series,
        pred_image_ids: pd.Series,
        gt_df: pd.DataFrame,
        pred_df: pd.DataFrame,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Matches one annotator-image entry between prediction and GT and computes
    the overlap matrices between its GT and predicted filaments.

    Args:
        annotator_image: Annotator-image identifier in the form ``"<annotator_id>-<image_id>"``.
        gt_annotator_image_ids: Series of annotator-image identifiers, one per row of ``gt_df``.
        pred_image_ids: Series of image identifiers, one per row of ``pred_df``.
        gt_df: Ground-truth filaments; columns ``filament_id`` and ``segmentation_rle``.
        pred_df: Predicted filaments; same columns as ``gt_df``.

    Returns:
        Tuple ``(iou_matrix, dice_matrix, n_gt, n_pred)`` where the matrices
        have shape (n_gt, n_pred).
    """
    image_id = annotator_image.split("-", maxsplit=1)[1]

    # (n_gt, H, W) stack with one layer per GT filament of this annotator-image
    gt_rles = gt_df.loc[gt_annotator_image_ids == annotator_image, "segmentation_rle"].tolist()
    gt_layers = torch.from_numpy(rles_to_layers(gt_rles))

    # (n_pred, H, W) stack with one layer per predicted filament of this image
    pred_rles = pred_df.loc[pred_image_ids == image_id, "segmentation_rle"].tolist()
    pred_layers = torch.from_numpy(rles_to_layers(pred_rles))

    n_gt = len(gt_rles)
    n_pred = len(pred_rles)

    iou_matrix, dice_matrix = get_overlap_matrices(gt_layers, pred_layers)

    return iou_matrix, dice_matrix, n_gt, n_pred

def get_overlap_df(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-annotator-image table of GT vs. prediction overlap matrices.

    Args:
        gt_df: Ground-truth filaments; columns ``filament_id`` and
            ``segmentation_rle``.
        pred_df: Predicted filaments; same columns as ``gt_df``.

    Returns:
        DataFrame with one row per annotator-image and columns:

        - ``annotator_image``: annotator-image identifier
        - ``iou_matrix``: IoU tensor of shape (n_gt, n_pred)
        - ``dice_matrix``: Dice tensor of shape (n_gt, n_pred)
        - ``n_gt``: number of GT filaments in the image
        - ``n_pred``: number of predicted filaments in the image
    """
    gt_annotator_image_ids = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_image_ids = pred_df["filament_id"].str.split("_", n=1).str[0]

    annotator_images = gt_annotator_image_ids.unique()

    overlap_df = pd.DataFrame({"annotator_image": annotator_images})

    results = overlap_df["annotator_image"].apply(
        process_entry,
        gt_annotator_image_ids=gt_annotator_image_ids,
        pred_image_ids=pred_image_ids,
        gt_df=gt_df,
        pred_df=pred_df,
    )

    overlap_df["iou_matrix"], overlap_df["dice_matrix"], overlap_df["n_gt"], overlap_df[
            "n_pred"] = zip(*results)

    return overlap_df

def get_pq_score(overlap_df: pd.DataFrame) -> float:
    """
    Compute the Panoptic Quality (PQ) score over all images.

    A GT/prediction pair is a true positive (TP) when its IoU exceeds the
    IoU threshold (0.5). Predictions with no match are false positives (FP),
    GT filaments with no match are false negatives (FN), and:

        PQ = sum(IoU of TP pairs) / (|TP| + 0.5 * |FP| + 0.5 * |FN|)

    Args:
        overlap_df: Per-image overlap table as produced by ``get_overlap_df``;
            must contain columns ``iou_matrix`` (IoU tensor of shape
            (n_gt, n_pred)), ``n_gt`` (GT filament count), and ``n_pred``
            (predicted filament count).

    Returns:
        Panoptic Quality score in ``[0, 1]`` (higher is better), or ``0.0`` if
        the denominator is zero.
    """
    iou_threshold: float = 0.5

    tp_iou_scores: list[float] = []
    fp_count = 0
    fn_count = 0

    for row in overlap_df.itertuples(index=False):
        iou_matrix = row.iou_matrix
        n_gt = row.n_gt
        n_pred = row.n_pred

        if n_gt == 0:
            fp_count += n_pred
            continue

        if n_pred == 0:
            fn_count += n_gt
            continue

        hit_matrix = iou_matrix > iou_threshold

        tp_iou_scores.extend(iou_matrix[hit_matrix].tolist())

        fp_count += fp_count_hit(hit_matrix)
        fn_count += fn_count_hit(hit_matrix)

    tp_count = len(tp_iou_scores)
    denominator = tp_count + 0.5 * fp_count + 0.5 * fn_count

    if denominator > 0:
        pq_score = sum(tp_iou_scores) / denominator
    else:
        pq_score = 0.0

    return pq_score

def plot_distribution(matrices: pd.Series, title: str, x_label: str):
    """
    Plots histogram of pairwise overlap scores between GT and predicted filaments
    (scores > 0 only).

    Flattens all per-image score matrices into a single sample, drops
    non-overlapping (zero) pairs, and plots normalized bar frequencies with
    a mean line.

    Args:
        matrices: Iterable of per-image score matrices, e.g. the ``iou_matrix``
            or ``dice_matrix`` column of the dataframe returned by
            ``get_overlap_df``.
        title: Plot title.
        x_label: X-axis label.
    """

    # every entry is an (n_gt, n_pred) tensor with its own shape, so they cannot be stacked
    scores = np.concatenate(
        [np.asarray(m, dtype=float).ravel() for m in matrices] or [np.zeros(0)]
    )
    # non-overlapping pairs carry no information and would swamp the zero bin
    scores = scores[scores > 0]

    x_max = 1.0 if scores.size == 0 or scores.max() <= 1.0 else scores.max()

    if x_max <= 1.0:
        n_bins = 50
        bins = np.linspace(0.0, x_max, n_bins + 1)
        bar_width = 0.02
    else:
        bins = np.arange(0, int(x_max) + 2)
        bar_width = 0.8

    counts, edges = np.histogram(scores, bins=bins)
    probabilities = counts / counts.sum() if counts.sum() > 0 else counts
    centers = (edges[:-1] + edges[1:]) / 2.0

    plt.bar(centers, probabilities, width=bar_width, edgecolor="black", alpha=0.7, color="navy")
    plt.grid(axis="y", linestyle="--")

    if scores.size > 0:
        mean_score = scores.mean()
        plt.axvline(
            mean_score,
            color="crimson",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean = {mean_score:.3f}",
        )
        plt.legend(fontsize=8)
    plt.xlabel(x_label)
    plt.ylabel("Probability")
    plt.title(title)


def plot_m2n_counts(overlap_df: pd.DataFrame, title: str = "Many to One and One to Many Distribution"):
    """
    Plots frequency of GT filaments and predictions matching multiple partners (IoU > 0).

    For each image, counts overlapping GT→pred (1:n) and pred→GT (n:1) links,
    then draws a split bar chart (GT matches on the right, prediction matches
    on the left).

    Args:
        overlap_df: Per-image overlap table as produced by ``get_overlap_df``;
            must contain columns ``iou_matrix``, ``n_gt``, and ``n_pred``.
        title: Plot title.
    """
    offset = 0.5

    # per-GT number of matched predictions (1:n) and per-prediction number of matched GTs (n:1)
    gt_match_counts: list[int] = []
    pred_match_counts: list[int] = []

    for row in overlap_df.itertuples(index=False):
        iou_matrix = row.iou_matrix
        n_gt = row.n_gt
        n_pred = row.n_pred

        if n_gt == 0:
            pred_match_counts.extend([0] * n_pred)
            continue

        if n_pred == 0:
            gt_match_counts.extend([0] * n_gt)
            continue

        hit_matrix = iou_matrix > 0

        gt_match_counts.extend(hit_matrix.sum(dim=1).tolist())
        pred_match_counts.extend(hit_matrix.sum(dim=0).tolist())

    gt_degree_counts = Counter(gt_match_counts)
    pred_degree_counts = Counter(pred_match_counts)

    gt_miss_count = gt_degree_counts.get(0, 0)
    gt_hit_count = sum(count for degree, count in gt_degree_counts.items() if degree > 0)
    pred_miss_count = pred_degree_counts.get(0, 0)
    pred_hit_count = sum(count for degree, count in pred_degree_counts.items() if degree > 0)

    color_gt_miss = "crimson"
    color_gt_hit = "forestgreen"
    color_pred_miss = "darkorange"
    color_pred_hit = "steelblue"

    gt_bars = {degree + offset: count for degree, count in gt_degree_counts.items()}
    pred_bars = {-(degree + offset): count for degree, count in pred_degree_counts.items()}

    gt_positions = sorted(gt_bars)
    pred_positions = sorted(pred_bars)
    bar_width = 0.8

    for pos in gt_positions:
        degree = int(round(pos - offset))
        plt.bar(
            pos,
            gt_bars[pos],
            width=bar_width,
            color=color_gt_miss if degree == 0 else color_gt_hit,
            edgecolor="black",
            alpha=0.7,
        )

    for pos in pred_positions:
        degree = int(round(-pos - offset))
        plt.bar(
            pos,
            pred_bars[pos],
            width=bar_width,
            color=color_pred_miss if degree == 0 else color_pred_hit,
            edgecolor="black",
            alpha=0.7,
        )

    plt.axvline(0, color="black", linewidth=0.8)
    plt.grid(axis="y", linestyle="--")
    plt.xlabel("Matching Case")
    plt.ylabel("Frequency")
    tick_positions = np.concatenate([gt_positions, pred_positions])
    tick_labels = [
        f"{abs(int(round(v + offset)))}:1" if v < 0 else f"1:{int(round(v - offset))}"
        for v in tick_positions
    ]
    plt.xticks(tick_positions, tick_labels, rotation=45, fontsize=9)
    plt.title(title)

    max_pos = max([abs(k) for k in gt_positions + pred_positions], default=offset)
    plt.xlim(-max_pos - bar_width, max_pos + bar_width)

    legend_handles = [
        Patch(facecolor=color_gt_hit, edgecolor="black", alpha=0.7, label=f"GT Hits (1:n, n>0): {gt_hit_count}"),
        Patch(facecolor=color_gt_miss, edgecolor="black", alpha=0.7, label=f"GT Misses (1:0): {gt_miss_count}"),
        Patch(facecolor=color_pred_miss, edgecolor="black", alpha=0.7, label=f"Prediction Misses (0:1): {pred_miss_count}"),
        Patch(facecolor=color_pred_hit, edgecolor="black", alpha=0.7, label=f"Prediction Hits (n:1, n>0): {pred_hit_count}"),
    ]
    plt.legend(handles=legend_handles, fontsize=9)

# plot_distribution(overlap_df["iou_matrix"], "IoU Distribution", "IoU")

# plot_distribution(overlap_df["dice_matrix"], "Dice Distribution", "Dice")
```
