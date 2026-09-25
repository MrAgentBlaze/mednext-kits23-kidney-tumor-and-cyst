25/09/2026 18:08
Device: cuda
GPU: NVIDIA GeForce RTX 4060 Ti
Training cases: 391
Validation cases: 98

============================================================
Starting MedNeXt-S training for 20 epochs
============================================================

Epoch 1/20
  Train loss: 0.704728
  Val loss:   0.613799
  Kidney Dice: 0.6352
  Tumor Dice:  0.0019
  Cyst Dice:   0.5306
  Mean Dice:   0.3892 <-- BEST

Epoch 2/20
  Train loss: 0.608177
  Val loss:   0.560124
  Kidney Dice: 0.7322
  Tumor Dice:  0.0061
  Cyst Dice:   0.5510
  Mean Dice:   0.4298 <-- BEST

Epoch 3/20
  Train loss: 0.569030
  Val loss:   0.547756
  Kidney Dice: 0.7664
  Tumor Dice:  0.0074
  Cyst Dice:   0.4898
  Mean Dice:   0.4212

Epoch 4/20
  Train loss: 0.559737
  Val loss:   0.531791
  Kidney Dice: 0.7710
  Tumor Dice:  0.1194
  Cyst Dice:   0.5102
  Mean Dice:   0.4669 <-- BEST

Epoch 5/20
  Train loss: 0.521147
  Val loss:   0.515641
  Kidney Dice: 0.7967
  Tumor Dice:  0.0825
  Cyst Dice:   0.2449
  Mean Dice:   0.3747

Epoch 6/20
  Train loss: 0.515539
  Val loss:   0.513202
  Kidney Dice: 0.7923
  Tumor Dice:  0.1247
  Cyst Dice:   0.2245
  Mean Dice:   0.3805

Epoch 7/20
  Train loss: 0.512388
  Val loss:   0.496588
  Kidney Dice: 0.7906
  Tumor Dice:  0.1647
  Cyst Dice:   0.1429
  Mean Dice:   0.3661

Epoch 8/20
  Train loss: 0.488209
  Val loss:   0.486061
  Kidney Dice: 0.8054
  Tumor Dice:  0.2275
  Cyst Dice:   0.0002
  Mean Dice:   0.3444

Epoch 9/20
  Train loss: 0.477848
  Val loss:   0.487641
  Kidney Dice: 0.8087
  Tumor Dice:  0.2354
  Cyst Dice:   0.0223
  Mean Dice:   0.3555

Epoch 10/20
  Train loss: 0.466887
  Val loss:   0.478180
  Kidney Dice: 0.8146
  Tumor Dice:  0.2032
  Cyst Dice:   0.0617
  Mean Dice:   0.3598

Epoch 11/20
  Train loss: 0.462319
  Val loss:   0.453377
  Kidney Dice: 0.8300
  Tumor Dice:  0.2471
  Cyst Dice:   0.0099
  Mean Dice:   0.3623

Epoch 12/20
  Train loss: 0.454328
  Val loss:   0.439899
  Kidney Dice: 0.8532
  Tumor Dice:  0.2782
  Cyst Dice:   0.0315
  Mean Dice:   0.3876

Epoch 13/20
  Train loss: 0.441577
  Val loss:   0.472291
  Kidney Dice: 0.8393
  Tumor Dice:  0.2550
  Cyst Dice:   0.0229
  Mean Dice:   0.3724

Epoch 14/20
  Train loss: 0.446463
  Val loss:   0.437744
  Kidney Dice: 0.8256
  Tumor Dice:  0.3400
  Cyst Dice:   0.0181
  Mean Dice:   0.3946

Epoch 15/20
  Train loss: 0.424819
  Val loss:   0.466426
  Kidney Dice: 0.8311
  Tumor Dice:  0.2209
  Cyst Dice:   0.0501
  Mean Dice:   0.3674

Epoch 16/20
  Train loss: 0.419538
  Val loss:   0.435388
  Kidney Dice: 0.8694
  Tumor Dice:  0.2522
  Cyst Dice:   0.0434
  Mean Dice:   0.3883

Epoch 17/20
  Train loss: 0.424755
  Val loss:   0.434117
  Kidney Dice: 0.8314
  Tumor Dice:  0.3401
  Cyst Dice:   0.0223
  Mean Dice:   0.3979

Epoch 18/20
  Train loss: 0.429474
  Val loss:   0.416083
  Kidney Dice: 0.8550
  Tumor Dice:  0.3326
  Cyst Dice:   0.0492
  Mean Dice:   0.4122

Epoch 19/20
  Train loss: 0.398347
  Val loss:   0.408051
  Kidney Dice: 0.8680
  Tumor Dice:  0.2701
  Cyst Dice:   0.0337
  Mean Dice:   0.3906

Epoch 20/20
  Train loss: 0.429048
  Val loss:   0.418394
  Kidney Dice: 0.8450
  Tumor Dice:  0.3648
  Cyst Dice:   0.0502
  Mean Dice:   0.4200

============================================================
Training completed.
============================================================
Best mean foreground Dice: 0.4669
Best model: outputs\mednext_s\best_model.pth
Last model: outputs\mednext_s\last_model.pth
History: outputs\mednext_s\training_history.csv