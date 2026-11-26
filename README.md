# ThyroHybridNet

One network predicts the thyroid gland and the nodule. The gland multiplicatively gates the nodule at every decoder scale. The nodule can only boost the gland. Small nodules are up-weighted in training, and nodule inference averages four scales.

Reported mean Dice is TN3K 0.872, TG3K gland 0.989, DDTI 0.722, and hospital gland 0.887 / nodule 0.807. A number printed on a frame is that frame's Dice. Green is the reference. Red is the prediction.

## TN3K nodules

![TN3K nodules across area bins](results/tn3k_by_area.png)

![1 to 5k pixels, Dice 0.692](results/tn3k_1to5k.png)

![5 to 15k pixels, Dice 0.829](results/tn3k_5to15k.png)

![15 to 40k pixels, Dice 0.858](results/tn3k_15to40k.png)

![Above 40k pixels, Dice 0.925](results/tn3k_over40k.png)

![TN3K Dice by nodule area](results/tn3k_area_dice.png)

![1–5k failures: 14 over-segmentation, 8 boundary mismatch, 1 miss](results/failure_1to5k.png)

## Hospital cohort

The left prediction is the public checkpoint on the same frame. The right prediction is the hospital-trained model.

![Public checkpoint and hospital-trained predictions](results/hospital_public_vs_ours.png)

![Nodule, 1 to 5k pixels](results/hospital_nodule_1to5k.png)

![Nodule, above 40k pixels](results/hospital_nodule_over40k.png)

![Gland, 1 to 5k pixels](results/hospital_gland_1to5k.png)

![Gland, above 40k pixels](results/hospital_gland_over40k.png)

## Gland and nodule together

![TN3K nodules and hospital gland frames](results/tn3k_and_hospital_gland.png)

## Network

![Shared encoder, two decoders, AGNG at every scale](results/architecture.png)

`code/published.py` is the published inference recipe: asymmetric guidance, inverse-root area weights, four-scale flip averaging, and the 15×15 soft gland gate. Training code is not included.
