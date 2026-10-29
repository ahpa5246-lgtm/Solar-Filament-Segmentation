# Submitted notebook audit and immediate Kaggle experiments

The audited input is existing confirmation notebook v7 (submitted file SHA256 b9d6dbba8d8c9a82e44a5d3dcd9eb361aa4068529273715d8b8e4c554640174e; notebook SHA256 c28ac0e81063cc7fbc8dcc839bced5cacb229932d0026d5b634dfb9f56e911a8). Kaggle submission 57747563 officially scored 0.32, tying the earlier best. No >=0.50 result exists at this audit boundary.

## Verified facts and fixes

1. Original .85 preprocessing/inference reconstructed the scored original masks with parity 1.0 over all 180 test observations. All 1,434 candidate .65 RLE rows passed native-mask round trips and official scoring. No evidence supports blaming a corrupt export, wrong orientation, duplicate IDs or an omitted test image.
2. TorchVision `maskrcnn_inference` returns sigmoid probabilities; another sigmoid/logit conversion in this inference path would be wrong. Grayscale is already divided by 255 and replicated into three channels.
3. Full continued training completed 16 epochs, reducing loss .7161 -> .6452, but never beat epoch-zero tuning PQ .40503. This is an objective/generalization mismatch, not a successful model improvement.
4. A real reconstruction mismatch was verified against TorchVision: `weights=None` defaults to five trainable backbone stages; the original pretrained factory defaults to three. Parameter `requires_grad` and module train/eval flags are not preserved by `state_dict`. The old continuation therefore silently unfreezes early backbone layers. `restore_training_contract` restores stages layer2/3/4 and freezes early stages. It also freezes running BN moments for batch-one continuation; the latter is an explicit modeling hypothesis, not a proven Public gain. A real CPU forward/backward test verifies frozen moments, unchanged counters and nonzero head gradients.
5. The interpolation notebook actually failed because it treated an updated `num_batches_tracked` counter as architecture drift. The regression reproduced the failure. The fix preserves original integral BN counters while still rejecting unrelated nonfloating state changes. A separate option restores original running mean/variance while retaining trained affine/model weights. Repaired interpolation is running as existing interpolation notebook v2.
6. Earlier NMS/none, NMS/NMS and orientation-aware union variants all lost confirmation PQ; they are not promoted. Lower crop threshold .65 tied official .32. These are not repeated as assumed fixes.
7. Detector test-time augmentation was absent; previous four-flip experiments affected the crop refiner. New code inverts detector flips before mask-IoU matching and requires support from distinct views. A single-view duplicate cannot impersonate multi-view agreement. Rectangular inverse-flip, crossing-mask and duplicate-vote regression tests pass. Existing confirmation notebook v8 evaluates original, base-preserving augmentation and two-view consensus on matched tuning/confirmation dates before optional final-weight inference.

## Open-source implementations actually reviewed

- TorchVision ROI heads: https://github.com/pytorch/vision/blob/main/torchvision/models/detection/roi_heads.py — mask probabilities, ROI target handling and inference semantics.
- TorchVision backbone utilities: https://github.com/pytorch/vision/blob/main/torchvision/models/detection/backbone_utils.py — trainable-stage defaults and the need to restore trainability after checkpoint loading. Also verified against the installed implementation.
- Weighted Boxes Fusion: https://github.com/ZFTurbo/Weighted-Boxes-Fusion/blob/master/ensemble_boxes/ensemble_boxes_wbf.py — matching detections and accounting for evidence from multiple views/models. New mask fusion code is independently implemented; no upstream source was copied into the project.
- ttach transformations: https://github.com/qubvel/ttach/blob/master/ttach/transforms.py — explicit image augmentation and inverse mask transforms.

## External AI model actually executed

Qwen/Qwen2.5-Coder-1.5B-Instruct, immutable model revision 2e1fd397ee46e1388853d2af2c993145b0f1098a, ran on Kaggle audit notebook v3. Input: 8,511 code/context tokens; output: 1,100 tokens; elapsed 126.43 seconds. No competition images/labels or credentials were sent to a model service. Open weights were downloaded and executed within Kaggle. The raw review, source hash and evidence are in EXTERNAL_AI_NOTEBOOK_REVIEW_20261028.json.

The small external model produced unsupported claims, including a nonexistent file path and assertions that grayscale replication, RLE guards and checkpoint hashes were missing. Those claims were checked and rejected. Threshold/NMS recommendations that repeated failed experiments were also rejected. Generated review text is advice, not measured evidence. Historical validation reuse and the loss-to-PQ mismatch remain real limitations.

## Active immutable execution identities

- External review: minahussain111/solar-filament-research-v6-audit/v3 — COMPLETE.
- Detector consensus: minahussain111/solar-filament-research-v6-confirmation/v8 — running; ledger DETECTOR_CONSENSUS_EXECUTION_20261028.json.
- BN recovery/interpolation: minahussain111/solar-v5-weight-interpolation-pq50/v2 — running; ledger V5_BN_RECOVERY_EXECUTION_20261028.json. Prior v1 terminal failure is preserved.
- Corrected four-epoch continuation: built and CPU-verified; wait for an existing GPU evaluation to finish before launching in the existing topology notebook. Do not duplicate active runs.

Full local suite after these fixes: 205 passed, one CUDA-only skipped. This verifies software behavior, not Public PQ. Preserve the original best and submit a unique validated candidate only after actual Kaggle execution. Any official score must come from a completed competition submission.
