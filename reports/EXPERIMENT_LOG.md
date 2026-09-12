# Experiment log - Altur acoustic detector

One record per experiment, appended by the scripts (never edited by hand; newest at the bottom).

## 2026-09-12 00:01:11 - shortcut check (trivial statistics, no voice content)

- **model**: logreg / GBM on call statistics
- **dataset_fraction**: 1.0000
- **segmentation**: n/a
- **selected_layer**: n/a
- **classifier**: logreg C=1 balanced; GBM 150x depth-2
- **hyperparameters**: default
- **device**: cpu
- **runtime_s**: seconds
- **best**: all_trivial/logreg: val AUC 1.000, acc 0.986
- **notes**: NOT a production model; measures dataset shortcuts

## 2026-09-12 00:01:54 - embedding extraction - wavlm

- **model**: microsoft/wavlm-base-plus
- **fraction**: 1.0000
- **segmentation**: VAD + 4s chunks
- **device**: NVIDIA GeForce RTX 4080 Laptop GPU
- **chunks**: {'train': 4102, 'val': 977}
- **runtime_s**: 134.0000
- **backbone_s**: 108.3000
- **peak_vram_mb**: 1864
- **realtime_factor**: 122.8000
- **notes**: frozen backbone, mean+std pooling of every hidden layer, fp16 cache

## 2026-09-12 00:03:09 - embedding extraction - wav2vec2_spanish

- **model**: facebook/wav2vec2-base-es-voxpopuli-v2
- **fraction**: 1.0000
- **segmentation**: VAD + 4s chunks
- **device**: NVIDIA GeForce RTX 4080 Laptop GPU
- **chunks**: {'train': 4102, 'val': 977}
- **runtime_s**: 75.2000
- **backbone_s**: 59.3000
- **peak_vram_mb**: 1884
- **realtime_factor**: 224.3000
- **notes**: frozen backbone, mean+std pooling of every hidden layer, fp16 cache

## 2026-09-12 00:04:05 - frozen wavlm (mean pooling) - layer probe + logreg + MLP

- **model**: microsoft/wavlm-base-plus
- **dataset_fraction**: 1.0000
- **segmentation**: VAD + 4s chunks, RMS -26.0 dB
- **selected_layer**: 0
- **classifier**: logreg C=1.0; MLP 768-256-2
- **hyperparameters**: lr=0.001, wd=0.0001, batch=64, dropout=0.3, patience=15, seed=42
- **device**: cuda
- **runtime_s**: 6.3000
- **val_logreg**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0019 (n=71)
- **val_mlp**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0000 (n=71)
- **notes**: MLP best epoch 4; aggregation = mean of chunk log-odds

## 2026-09-12 00:04:18 - frozen wav2vec2_spanish (mean pooling) - layer probe + logreg + MLP

- **model**: facebook/wav2vec2-base-es-voxpopuli-v2
- **dataset_fraction**: 1.0000
- **segmentation**: VAD + 4s chunks, RMS -26.0 dB
- **selected_layer**: 0
- **classifier**: logreg C=1.0; MLP 768-256-2
- **hyperparameters**: lr=0.001, wd=0.0001, batch=64, dropout=0.3, patience=15, seed=42
- **device**: cuda
- **runtime_s**: 6.2000
- **val_logreg**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0008 (n=71)
- **val_mlp**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0005 (n=71)
- **notes**: MLP best epoch 4; aggregation = mean of chunk log-odds

## 2026-09-12 00:05:26 - embedding extraction - xlsr

- **model**: facebook/wav2vec2-xls-r-300m
- **fraction**: 1.0000
- **segmentation**: VAD + 4s chunks
- **device**: NVIDIA GeForce RTX 4080 Laptop GPU
- **chunks**: {'train': 4102, 'val': 977}
- **runtime_s**: 136.3000
- **backbone_s**: 109.3000
- **peak_vram_mb**: 2996
- **realtime_factor**: 121.7000
- **notes**: frozen backbone, mean+std pooling of every hidden layer, fp16 cache

## 2026-09-12 00:06:56 - frozen xlsr (mean pooling) - layer probe + logreg + MLP

- **model**: facebook/wav2vec2-xls-r-300m
- **dataset_fraction**: 1.0000
- **segmentation**: VAD + 4s chunks, RMS -26.0 dB
- **selected_layer**: 0
- **classifier**: logreg C=1.0; MLP 1024-256-2
- **hyperparameters**: lr=0.001, wd=0.0001, batch=64, dropout=0.3, patience=15, seed=42
- **device**: cuda
- **runtime_s**: 8.6000
- **val_logreg**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0000 (n=71)
- **val_mlp**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0000 (n=71)
- **notes**: MLP best epoch 15; aggregation = mean of chunk log-odds

## 2026-09-12 00:13:17 - fine-tuning wavlm

- **model**: microsoft/wavlm-base-plus
- **dataset_fraction**: 0.0500
- **segmentation**: as frozen pipeline
- **selected_layer**: 0
- **classifier**: MLP head (warm start)
- **hyperparameters**: unfreeze=4, head_epochs=1, unfrozen_epochs=1, batch=8, backbone_lr=1e-05, head_lr=0.0001
- **device**: cuda
- **runtime_s**: 29
- **frozen_val**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0000 (n=71)
- **finetuned_val**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0000 (n=4)
- **winner**: frozen
- **best_epoch**: 2

## 2026-09-12 00:19:43 - stress-test layer probe - wavlm

- **model**: microsoft/wavlm-base-plus
- **dataset_fraction**: 1.0000
- **segmentation**: as pipeline; validation caller channel perturbed
- **selected_layer**: 9
- **classifier**: logreg per layer (clean train)
- **hyperparameters**: conditions=['clean', 'gain_-12dB', 'lowpass_3400', 'lowpass_3000', 'pstn_300-3400', 'white_snr20', 'white_snr10', 'pink_snr10', 'mulaw_codec', 'reverb_0.3s', 'tilt_-3dB_oct']
- **device**: gpu (extraction) / cpu (probe)
- **runtime_s**: n/a
- **mean_auc**: 0.9990
- **worst_auc**: 0.9905
- **notes**: clean validation saturated (AUC 1.0 on every layer); layer chosen on stressed validation

## 2026-09-12 00:19:46 - stress-test layer probe - wav2vec2_spanish

- **model**: facebook/wav2vec2-base-es-voxpopuli-v2
- **dataset_fraction**: 1.0000
- **segmentation**: as pipeline; validation caller channel perturbed
- **selected_layer**: 5
- **classifier**: logreg per layer (clean train)
- **hyperparameters**: conditions=['clean', 'gain_-12dB', 'lowpass_3400', 'lowpass_3000', 'pstn_300-3400', 'white_snr20', 'white_snr10', 'pink_snr10', 'mulaw_codec', 'reverb_0.3s', 'tilt_-3dB_oct']
- **device**: gpu (extraction) / cpu (probe)
- **runtime_s**: n/a
- **mean_auc**: 0.9999
- **worst_auc**: 0.9992
- **notes**: clean validation saturated (AUC 1.0 on every layer); layer chosen on stressed validation

## 2026-09-12 00:19:53 - stress-test layer probe - xlsr

- **model**: facebook/wav2vec2-xls-r-300m
- **dataset_fraction**: 1.0000
- **segmentation**: as pipeline; validation caller channel perturbed
- **selected_layer**: 23
- **classifier**: logreg per layer (clean train)
- **hyperparameters**: conditions=['clean', 'gain_-12dB', 'lowpass_3400', 'lowpass_3000', 'pstn_300-3400', 'white_snr20', 'white_snr10', 'pink_snr10', 'mulaw_codec', 'reverb_0.3s', 'tilt_-3dB_oct']
- **device**: gpu (extraction) / cpu (probe)
- **runtime_s**: n/a
- **mean_auc**: 0.9993
- **worst_auc**: 0.9960
- **notes**: clean validation saturated (AUC 1.0 on every layer); layer chosen on stressed validation

## 2026-09-12 00:20:13 - frozen wavlm (mean pooling) - layer probe + logreg + MLP

- **model**: microsoft/wavlm-base-plus
- **dataset_fraction**: 1.0000
- **segmentation**: VAD + 4s chunks, RMS -26.0 dB
- **selected_layer**: 9
- **classifier**: logreg C=1.0; MLP 768-256-2
- **hyperparameters**: lr=0.001, wd=0.0001, batch=64, dropout=0.3, patience=15, seed=42
- **device**: cuda
- **runtime_s**: 4.5000
- **val_logreg**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0000 (n=71)
- **val_mlp**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0000 (n=71)
- **notes**: MLP best epoch 3; aggregation = mean of chunk log-odds

## 2026-09-12 00:20:24 - frozen wav2vec2_spanish (mean pooling) - layer probe + logreg + MLP

- **model**: facebook/wav2vec2-base-es-voxpopuli-v2
- **dataset_fraction**: 1.0000
- **segmentation**: VAD + 4s chunks, RMS -26.0 dB
- **selected_layer**: 5
- **classifier**: logreg C=1.0; MLP 768-256-2
- **hyperparameters**: lr=0.001, wd=0.0001, batch=64, dropout=0.3, patience=15, seed=42
- **device**: cuda
- **runtime_s**: 4.8000
- **val_logreg**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0002 (n=71)
- **val_mlp**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0000 (n=71)
- **notes**: MLP best epoch 3; aggregation = mean of chunk log-odds

## 2026-09-12 00:20:37 - frozen xlsr (mean pooling) - layer probe + logreg + MLP

- **model**: facebook/wav2vec2-xls-r-300m
- **dataset_fraction**: 1.0000
- **segmentation**: VAD + 4s chunks, RMS -26.0 dB
- **selected_layer**: 23
- **classifier**: logreg C=1.0; MLP 1024-256-2
- **hyperparameters**: lr=0.001, wd=0.0001, batch=64, dropout=0.3, patience=15, seed=42
- **device**: cuda
- **runtime_s**: 6.1000
- **val_logreg**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0000 (n=71)
- **val_mlp**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0000 (n=71)
- **notes**: MLP best epoch 5; aggregation = mean of chunk log-odds

## 2026-09-12 00:20:42 - stress-test MLP - wavlm

- **model**: microsoft/wavlm-base-plus
- **dataset_fraction**: 1.0000
- **segmentation**: validation caller channel perturbed
- **selected_layer**: 9
- **classifier**: MLP (clean train)
- **hyperparameters**: conditions=['clean', 'gain_-12dB', 'lowpass_3400', 'lowpass_3000', 'pstn_300-3400', 'white_snr20', 'white_snr10', 'pink_snr10', 'mulaw_codec', 'reverb_0.3s', 'tilt_-3dB_oct']
- **device**: cuda
- **runtime_s**: seconds
- **mean_stressed_auc**: 0.9967
- **mean_stressed_acc**: 0.9296
- **worst**: tilt_-3dB_oct AUC 0.976

## 2026-09-12 00:20:43 - stress-test MLP - wav2vec2_spanish

- **model**: facebook/wav2vec2-base-es-voxpopuli-v2
- **dataset_fraction**: 1.0000
- **segmentation**: validation caller channel perturbed
- **selected_layer**: 5
- **classifier**: MLP (clean train)
- **hyperparameters**: conditions=['clean', 'gain_-12dB', 'lowpass_3400', 'lowpass_3000', 'pstn_300-3400', 'white_snr20', 'white_snr10', 'pink_snr10', 'mulaw_codec', 'reverb_0.3s', 'tilt_-3dB_oct']
- **device**: cuda
- **runtime_s**: seconds
- **mean_stressed_auc**: 0.9998
- **mean_stressed_acc**: 0.9718
- **worst**: white_snr10 AUC 0.998

## 2026-09-12 00:20:44 - stress-test MLP - xlsr

- **model**: facebook/wav2vec2-xls-r-300m
- **dataset_fraction**: 1.0000
- **segmentation**: validation caller channel perturbed
- **selected_layer**: 23
- **classifier**: MLP (clean train)
- **hyperparameters**: conditions=['clean', 'gain_-12dB', 'lowpass_3400', 'lowpass_3000', 'pstn_300-3400', 'white_snr20', 'white_snr10', 'pink_snr10', 'mulaw_codec', 'reverb_0.3s', 'tilt_-3dB_oct']
- **device**: cuda
- **runtime_s**: seconds
- **mean_stressed_auc**: 0.9975
- **mean_stressed_acc**: 0.8620
- **worst**: white_snr10 AUC 0.980

## 2026-09-12 00:20:47 - calibration - wavlm

- **model**: microsoft/wavlm-base-plus
- **dataset_fraction**: val only
- **segmentation**: as trained
- **selected_layer**: as trained
- **classifier**: Platt / temperature on call score
- **hyperparameters**: 5-fold CV on validation
- **device**: cpu
- **runtime_s**: seconds
- **mlp**: raw Brier 0.0000 -> temperature Brier 0.0000
- **notes**: calibrator refitted on all validation calls for deployment

## 2026-09-12 00:20:47 - calibration - wav2vec2_spanish

- **model**: facebook/wav2vec2-base-es-voxpopuli-v2
- **dataset_fraction**: val only
- **segmentation**: as trained
- **selected_layer**: as trained
- **classifier**: Platt / temperature on call score
- **hyperparameters**: 5-fold CV on validation
- **device**: cpu
- **runtime_s**: seconds
- **mlp**: raw Brier 0.0000 -> temperature Brier 0.0000
- **notes**: calibrator refitted on all validation calls for deployment

## 2026-09-12 00:20:47 - calibration - xlsr

- **model**: facebook/wav2vec2-xls-r-300m
- **dataset_fraction**: val only
- **segmentation**: as trained
- **selected_layer**: as trained
- **classifier**: Platt / temperature on call score
- **hyperparameters**: 5-fold CV on validation
- **device**: cpu
- **runtime_s**: seconds
- **mlp**: raw Brier 0.0000 -> temperature Brier 0.0000
- **notes**: calibrator refitted on all validation calls for deployment

## 2026-09-12 00:21:34 - benchmark: three frozen backbones

- **model**: microsoft/wavlm-base-plus, facebook/wav2vec2-base-es-voxpopuli-v2, facebook/wav2vec2-xls-r-300m
- **dataset_fraction**: 1.0000
- **segmentation**: shared
- **selected_layer**: {'wavlm': 9, 'wav2vec2_spanish': 5, 'xlsr': 23}
- **classifier**: MLP (same for all)
- **hyperparameters**: see train_classifier
- **device**: cuda
- **runtime_s**: n/a
- **winner**: wav2vec2_spanish layer 5
- **reason**: highest mean validation AUC under the 10 stress conditions (0.9998; stressed accuracy 0.972, worst condition white_snr10 AUC 0.998); clean validation accuracy 1.000 / AUC 1.0000 - the clean split is saturated for every backbone and cannot rank them; 3 model(s) within 0.005 -> tie-break by stressed accuracy, EER, latency
- **val_auc**: {'wavlm': 1.0, 'wav2vec2_spanish': 1.0, 'xlsr': 1.0}
- **val_accuracy**: {'wavlm': 1.0, 'wav2vec2_spanish': 1.0, 'xlsr': 1.0}
- **stress_mean_auc**: {'wavlm': 0.9967, 'wav2vec2_spanish': 0.9998, 'xlsr': 0.9975}

## 2026-09-12 00:24:01 - calibration - wav2vec2_spanish

- **model**: facebook/wav2vec2-base-es-voxpopuli-v2
- **dataset_fraction**: val only
- **segmentation**: as trained
- **selected_layer**: as trained
- **classifier**: Platt / temperature on call score
- **hyperparameters**: 5-fold CV on validation
- **device**: cpu
- **runtime_s**: seconds
- **mlp**: raw Brier 0.0200 -> platt Brier 0.0196
- **notes**: calibrator refitted on all validation calls for deployment

## 2026-09-12 00:38:26 - calibration - wavlm

- **model**: microsoft/wavlm-base-plus
- **dataset_fraction**: val only
- **segmentation**: as trained
- **selected_layer**: as trained
- **classifier**: Platt / temperature on call score
- **hyperparameters**: 5-fold CV on validation
- **device**: cpu
- **runtime_s**: seconds
- **mlp**: raw Brier 0.0597 -> temperature Brier 0.0565
- **notes**: calibrator refitted on all validation calls for deployment

## 2026-09-12 00:38:27 - calibration - wav2vec2_spanish

- **model**: facebook/wav2vec2-base-es-voxpopuli-v2
- **dataset_fraction**: val only
- **segmentation**: as trained
- **selected_layer**: as trained
- **classifier**: Platt / temperature on call score
- **hyperparameters**: 5-fold CV on validation
- **device**: cpu
- **runtime_s**: seconds
- **mlp**: raw Brier 0.0200 -> platt Brier 0.0196
- **notes**: calibrator refitted on all validation calls for deployment

## 2026-09-12 00:38:27 - calibration - xlsr

- **model**: facebook/wav2vec2-xls-r-300m
- **dataset_fraction**: val only
- **segmentation**: as trained
- **selected_layer**: as trained
- **classifier**: Platt / temperature on call score
- **hyperparameters**: 5-fold CV on validation
- **device**: cpu
- **runtime_s**: seconds
- **mlp**: raw Brier 0.1014 -> platt Brier 0.0771
- **notes**: calibrator refitted on all validation calls for deployment


## Runpod pod s1mv90hscq6vls (NVIDIA L4) - entries copied from the pod log
## 2026-09-12 06:40:39 - embedding extraction - wav2vec2_spanish

- **model**: facebook/wav2vec2-base-es-voxpopuli-v2
- **fraction**: 1.0000
- **segmentation**: VAD + 4s chunks
- **device**: NVIDIA L4
- **chunks**: {'train': 4102, 'val': 977}
- **runtime_s**: 63.1000
- **backbone_s**: 38.2000
- **peak_vram_mb**: 1443
- **realtime_factor**: 347.8000
- **notes**: frozen backbone, mean+std pooling of every hidden layer, fp16 cache

## 2026-09-12 06:49:28 - fine-tuning wav2vec2_spanish

- **model**: facebook/wav2vec2-base-es-voxpopuli-v2
- **dataset_fraction**: 1.0000
- **segmentation**: as frozen pipeline
- **selected_layer**: 5
- **classifier**: MLP head (warm start)
- **hyperparameters**: unfreeze=4, head_epochs=3, unfrozen_epochs=8, batch=8, backbone_lr=1e-05, head_lr=0.0001
- **device**: cuda
- **runtime_s**: 323
- **frozen_val**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0000 (n=71)
- **finetuned_val**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0000 (n=71)
- **winner**: frozen
- **best_epoch**: 4
- **rule**: mean stressed validation AUC (fine-tuned 0.9998 vs frozen 0.9998; fine-tuned must win by > 0.005)

## 2026-09-12 06:54:50 - augmentation experiment - wav2vec2_spanish

- **model**: facebook/wav2vec2-base-es-voxpopuli-v2
- **dataset_fraction**: 1.0000
- **segmentation**: as pipeline; train caller channel perturbed (seen families)
- **selected_layer**: 5
- **classifier**: MLP (same recipe), clean vs clean+augmented
- **hyperparameters**: seen=['gain_-12dB', 'white_snr20', 'white_snr10', 'pink_snr10', 'mulaw_codec']; unseen=['lowpass_3400', 'lowpass_3000', 'pstn_300-3400', 'reverb_0.3s', 'tilt_-3dB_oct']
- **device**: cuda
- **runtime_s**: 316
- **unseen_auc**: baseline 0.9998 -> augmented 1.0000
- **unseen_acc**: baseline 0.955 -> augmented 0.938
- **winner**: baseline

## 2026-09-12 01:34:25 - benchmark: three frozen backbones

- **model**: microsoft/wavlm-base-plus, facebook/wav2vec2-base-es-voxpopuli-v2, facebook/wav2vec2-xls-r-300m
- **dataset_fraction**: 1.0000
- **segmentation**: shared
- **selected_layer**: {'wavlm': 9, 'wav2vec2_spanish': 5, 'xlsr': 23}
- **classifier**: MLP (same for all)
- **hyperparameters**: see train_classifier
- **device**: cuda
- **runtime_s**: n/a
- **winner**: wav2vec2_spanish layer 5
- **reason**: highest mean validation AUC under the 10 stress conditions (0.9998; stressed accuracy 0.972, worst condition white_snr10 AUC 0.998); clean validation accuracy 1.000 / AUC 1.0000 - the clean split is saturated for every backbone and cannot rank them; 3 model(s) within 0.005 -> tie-break by stressed accuracy, EER, latency
- **val_auc**: {'wavlm': 1.0, 'wav2vec2_spanish': 1.0, 'xlsr': 1.0}
- **val_accuracy**: {'wavlm': 1.0, 'wav2vec2_spanish': 1.0, 'xlsr': 1.0}
- **stress_mean_auc**: {'wavlm': 0.9967, 'wav2vec2_spanish': 0.9998, 'xlsr': 0.9975}

## 2026-09-12 01:48:25 - sanity checks - wav2vec2_spanish

- **model**: facebook/wav2vec2-base-es-voxpopuli-v2
- **dataset_fraction**: 1.0000
- **segmentation**: cached embeddings
- **selected_layer**: 5
- **classifier**: logreg
- **hyperparameters**: permutation x5; clustering cosine/average
- **device**: cpu
- **runtime_s**: 2
- **permutation_auc**: [0.471, 0.532, 0.51, 0.394, 0.592]
- **one_chunk_accuracy**: 1.0000
- **voices**: synthetic k=2, human k=2
- **loo_synthetic**: {'min_held_out_accuracy': 0.17486338797814208, 'call_weighted_accuracy': 0.2561576354679803, 'clusters_below_0.9': 1}
- **loo_human**: {'min_held_out_accuracy': 0.10884353741496598, 'call_weighted_accuracy': 0.12666666666666668, 'clusters_below_0.9': 1}

## 2026-09-12 01:52:25 - voice identity + leave-one-voice-out

- **model**: microsoft/wavlm-base-plus-sv (measuring) / wav2vec2_spanish L5 (detector)
- **dataset_fraction**: 1.0000
- **segmentation**: caller chunks
- **selected_layer**: 5
- **classifier**: logreg
- **hyperparameters**: cosine threshold 0.7
- **device**: cuda
- **runtime_s**: 73
- **clusters**: {"0.6": {"synthetic": 2, "human": 2}, "0.65": {"synthetic": 3, "human": 2}, "0.7": {"synthetic": 3, "human": 2}, "0.75": {"synthetic": 3, "human": 2}, "0.8": {"synthetic": 5, "human": 3}, "0.86": {"synthetic": 6, "human": 4}}
- **loo**: {'threshold': 0.7, 'voices_tested': 2, 'calls_tested': 202, 'call_weighted_recognised': 0.8811881188118812, 'min_recognised': 0.8349514563106796, 'mean_auc': 0.9994752033586984}

## 2026-09-12 01:52:50 - voice identity + leave-one-voice-out

- **model**: microsoft/wavlm-base-plus-sv (measuring) / wav2vec2_spanish L5 (detector)
- **dataset_fraction**: 1.0000
- **segmentation**: caller chunks
- **selected_layer**: 5
- **classifier**: logreg
- **hyperparameters**: cosine threshold 0.86
- **device**: cuda
- **runtime_s**: 1
- **clusters**: {"0.6": {"synthetic": 2, "human": 2}, "0.65": {"synthetic": 3, "human": 2}, "0.7": {"synthetic": 3, "human": 2}, "0.75": {"synthetic": 3, "human": 2}, "0.8": {"synthetic": 5, "human": 3}, "0.86": {"synthetic": 6, "human": 4}}
- **loo**: {'threshold': 0.86, 'voices_tested': 4, 'calls_tested': 199, 'call_weighted_recognised': 0.9547738693467337, 'min_recognised': 0.7222222222222222, 'mean_auc': 1.0}

## 2026-09-12 02:03:15 - external out-of-dataset check

- **model**: wav2vec2_spanish layer 5 + mlp (deployed)
- **dataset_fraction**: n/a
- **segmentation**: as pipeline
- **selected_layer**: 5
- **classifier**: deployed
- **hyperparameters**: edge-tts/gTTS/SAPI vs FLEURS/ASVspoof bonafide
- **device**: cuda
- **runtime_s**: 91
- **results**: {'8k': {'accuracy': 0.685, 'balanced_accuracy': 0.713, 'auc': 0.9, 'eer': 0.167, 'far': 0.042, 'frr': 0.533}, 'pstn': {'accuracy': 0.88, 'balanced_accuracy': 0.875, 'auc': 0.945, 'eer': 0.129, 'far': 0.167, 'frr': 0.083}}

## 2026-09-12 03:17:30 - sanity checks - wav2vec2_spanish

- **model**: facebook/wav2vec2-base-es-voxpopuli-v2
- **dataset_fraction**: 1.0000
- **segmentation**: cached embeddings
- **selected_layer**: 5
- **classifier**: logreg
- **hyperparameters**: permutation x5; clustering cosine/average
- **device**: cpu
- **runtime_s**: 3
- **permutation_auc**: [0.471, 0.532, 0.51, 0.394, 0.592]
- **one_chunk_accuracy**: 1.0000
- **voices**: synthetic k=2, human k=2
- **loo_synthetic**: {'min_held_out_accuracy': 0.17486338797814208, 'call_weighted_accuracy': 0.2561576354679803, 'clusters_below_0.9': 1}
- **loo_human**: {'min_held_out_accuracy': 0.10884353741496598, 'call_weighted_accuracy': 0.12666666666666668, 'clusters_below_0.9': 1}

## 2026-09-12 07:15:42 - channel pilot - train/seen

- **model**: deployed specialist (Model A)
- **n_samples**: 10
- **accuracy_original**: 1.0000
- **accuracy_channel**: 0.7000
- **accuracy_channel_silero**: 0.7000
- **verdict_flips**: 3
- **segmentation_failures**: 4
- **mean_abs_delta_p**: 0.3393
- **decision**: GO
- **notes**: 10-call gate before generating the full channel dataset

## 2026-09-12 07:38:56 - channel pilot - train/seen

- **model**: deployed specialist (Model A)
- **n_samples**: 10
- **accuracy_original**: 1.0000
- **accuracy_channel**: 0.7000
- **accuracy_channel_silero**: 0.7000
- **verdict_flips**: 3
- **segmentation_failures**: 4
- **mean_abs_delta_p**: 0.3337
- **decision**: GO
- **notes**: 10-call gate before generating the full channel dataset

## 2026-09-12 07:40:02 - channel pilot - train/seen

- **model**: deployed specialist (Model A)
- **n_samples**: 10
- **accuracy_original**: 1.0000
- **accuracy_channel**: 0.7000
- **accuracy_channel_silero**: 0.7000
- **verdict_flips**: 3
- **segmentation_failures**: 4
- **mean_abs_delta_p**: 0.3297
- **decision**: GO
- **notes**: 10-call gate before generating the full channel dataset

## 2026-09-12 08:02:36 - embedding extraction - original + telephone channel

- **model**: facebook/wav2vec2-base-es-voxpopuli-v2
- **segmentation**: silero VAD + 4s chunks, RMS -26.0 dB
- **device**: NVIDIA GeForce RTX 4080 Laptop GPU
- **sets**: {'train_original': 3861, 'train_channel': 6528, 'val_original': 935, 'val_channel_seen': 815, 'val_channel_unseen': 821}
- **runtime_s**: 1337.2000
- **notes**: frozen backbone, mean pooling of every hidden layer; train sets are the only ones that may be trained on

## 2026-09-12 08:03:50 - Robust Acoustic V2 - orig_only

- **model**: facebook/wav2vec2-base-es-voxpopuli-v2 (frozen)
- **training_domains**: train_original
- **n_train_chunks**: 3861
- **selected_layer**: 1
- **segmentation**: silero VAD + 4s chunks, RMS -26 dB
- **classifier**: MLP 768-256-2
- **hyperparameters**: lr=0.001, wd=0.0001, batch=64, dropout=0.3, seed=42, best epoch 10
- **device**: cuda
- **runtime_s**: 4.6000
- **val_original**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0000 (n=71)
- **val_channel_seen**: acc=0.5493 bal_acc=0.5294 f1=0.1111 AUC=0.9316 EER=14.11% FAR=0.941 FRR=0.000 brier=0.4321 (n=71)
- **val_channel_unseen**: acc=0.5352 bal_acc=0.5147 f1=0.0571 AUC=0.9388 EER=14.11% FAR=0.971 FRR=0.000 brier=0.4550 (n=71)
- **notes**: control; channel_unseen never used for selection

## 2026-09-12 08:03:50 - Robust Acoustic V2 - channel_only

- **model**: facebook/wav2vec2-base-es-voxpopuli-v2 (frozen)
- **training_domains**: train_channel
- **n_train_chunks**: 6528
- **selected_layer**: 1
- **segmentation**: silero VAD + 4s chunks, RMS -26 dB
- **classifier**: MLP 768-256-2
- **hyperparameters**: lr=0.001, wd=0.0001, batch=64, dropout=0.3, seed=42, best epoch 10
- **device**: cuda
- **runtime_s**: 3.6000
- **val_original**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0002 (n=71)
- **val_channel_seen**: acc=0.9577 bal_acc=0.9595 f1=0.9577 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.081 brier=0.0390 (n=71)
- **val_channel_unseen**: acc=0.9577 bal_acc=0.9571 f1=0.9552 AUC=0.9944 EER=2.82% FAR=0.059 FRR=0.027 brier=0.0305 (n=71)
- **notes**: DEPLOYED as Model B; channel_unseen never used for selection

## 2026-09-12 08:03:50 - Robust Acoustic V2 - mixed

- **model**: facebook/wav2vec2-base-es-voxpopuli-v2 (frozen)
- **training_domains**: train_original, train_channel
- **n_train_chunks**: 10389
- **selected_layer**: 1
- **segmentation**: silero VAD + 4s chunks, RMS -26 dB
- **classifier**: MLP 768-256-2
- **hyperparameters**: lr=0.001, wd=0.0001, batch=64, dropout=0.3, seed=42, best epoch 8
- **device**: cuda
- **runtime_s**: 5.4000
- **val_original**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0000 (n=71)
- **val_channel_seen**: acc=0.9577 bal_acc=0.9595 f1=0.9577 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.081 brier=0.0361 (n=71)
- **val_channel_unseen**: acc=0.9718 bal_acc=0.9718 f1=0.9706 AUC=0.9952 EER=2.82% FAR=0.029 FRR=0.027 brier=0.0199 (n=71)
- **notes**: control; channel_unseen never used for selection

## 2026-09-12 08:03:50 - Robust Acoustic V2 - balanced

- **model**: facebook/wav2vec2-base-es-voxpopuli-v2 (frozen)
- **training_domains**: train_original, train_channel
- **n_train_chunks**: 10389
- **selected_layer**: 1
- **segmentation**: silero VAD + 4s chunks, RMS -26 dB
- **classifier**: MLP 768-256-2
- **hyperparameters**: lr=0.001, wd=0.0001, batch=64, dropout=0.3, seed=42, best epoch 18
- **device**: cuda
- **runtime_s**: 7.5000
- **val_original**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0000 (n=71)
- **val_channel_seen**: acc=0.9718 bal_acc=0.9730 f1=0.9714 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.054 brier=0.0262 (n=71)
- **val_channel_unseen**: acc=0.9718 bal_acc=0.9718 f1=0.9706 AUC=0.9968 EER=2.82% FAR=0.029 FRR=0.027 brier=0.0257 (n=71)
- **notes**: control; channel_unseen never used for selection

## 2026-09-12 08:05:28 - Robust Acoustic V2 - orig_only

- **model**: facebook/wav2vec2-base-es-voxpopuli-v2 (frozen)
- **training_domains**: train_original
- **n_train_chunks**: 3861
- **selected_layer**: 5
- **segmentation**: silero VAD + 4s chunks, RMS -26 dB
- **classifier**: MLP 768-256-2
- **hyperparameters**: lr=0.001, wd=0.0001, batch=64, dropout=0.3, seed=42, best epoch 1
- **device**: cuda
- **runtime_s**: 3.0000
- **val_original**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0002 (n=71)
- **val_channel_seen**: acc=0.9859 bal_acc=0.9853 f1=0.9851 AUC=1.0000 EER=0.00% FAR=0.029 FRR=0.000 brier=0.0203 (n=71)
- **val_channel_unseen**: acc=0.9296 bal_acc=0.9265 f1=0.9206 AUC=1.0000 EER=0.00% FAR=0.147 FRR=0.000 brier=0.0757 (n=71)
- **notes**: control; channel_unseen never used for selection

## 2026-09-12 08:05:28 - Robust Acoustic V2 - channel_only

- **model**: facebook/wav2vec2-base-es-voxpopuli-v2 (frozen)
- **training_domains**: train_channel
- **n_train_chunks**: 6528
- **selected_layer**: 5
- **segmentation**: silero VAD + 4s chunks, RMS -26 dB
- **classifier**: MLP 768-256-2
- **hyperparameters**: lr=0.001, wd=0.0001, batch=64, dropout=0.3, seed=42, best epoch 1
- **device**: cuda
- **runtime_s**: 2.3000
- **val_original**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0004 (n=71)
- **val_channel_seen**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0038 (n=71)
- **val_channel_unseen**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0061 (n=71)
- **notes**: DEPLOYED as Model B; channel_unseen never used for selection

## 2026-09-12 08:05:28 - Robust Acoustic V2 - mixed

- **model**: facebook/wav2vec2-base-es-voxpopuli-v2 (frozen)
- **training_domains**: train_original, train_channel
- **n_train_chunks**: 10389
- **selected_layer**: 5
- **segmentation**: silero VAD + 4s chunks, RMS -26 dB
- **classifier**: MLP 768-256-2
- **hyperparameters**: lr=0.001, wd=0.0001, batch=64, dropout=0.3, seed=42, best epoch 1
- **device**: cuda
- **runtime_s**: 3.6000
- **val_original**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0001 (n=71)
- **val_channel_seen**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0024 (n=71)
- **val_channel_unseen**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0024 (n=71)
- **notes**: control; channel_unseen never used for selection

## 2026-09-12 08:05:28 - Robust Acoustic V2 - balanced

- **model**: facebook/wav2vec2-base-es-voxpopuli-v2 (frozen)
- **training_domains**: train_original, train_channel
- **n_train_chunks**: 10389
- **selected_layer**: 5
- **segmentation**: silero VAD + 4s chunks, RMS -26 dB
- **classifier**: MLP 768-256-2
- **hyperparameters**: lr=0.001, wd=0.0001, batch=64, dropout=0.3, seed=42, best epoch 1
- **device**: cuda
- **runtime_s**: 3.7000
- **val_original**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0000 (n=71)
- **val_channel_seen**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0016 (n=71)
- **val_channel_unseen**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0030 (n=71)
- **notes**: control; channel_unseen never used for selection

## 2026-09-12 08:06:45 - Robust Acoustic V2 - orig_only

- **model**: facebook/wav2vec2-base-es-voxpopuli-v2 (frozen)
- **training_domains**: train_original
- **n_train_chunks**: 3861
- **selected_layer**: 5
- **segmentation**: silero VAD + 4s chunks, RMS -26 dB
- **classifier**: MLP 768-256-2
- **hyperparameters**: lr=0.001, wd=0.0001, batch=64, dropout=0.3, seed=42, best epoch 1
- **device**: cuda
- **runtime_s**: 3.0000
- **val_original**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0002 (n=71)
- **val_channel_seen**: acc=0.9859 bal_acc=0.9853 f1=0.9851 AUC=1.0000 EER=0.00% FAR=0.029 FRR=0.000 brier=0.0203 (n=71)
- **val_channel_unseen**: acc=0.9296 bal_acc=0.9265 f1=0.9206 AUC=1.0000 EER=0.00% FAR=0.147 FRR=0.000 brier=0.0757 (n=71)
- **notes**: control; channel_unseen never used for selection

## 2026-09-12 08:06:45 - Robust Acoustic V2 - channel_only

- **model**: facebook/wav2vec2-base-es-voxpopuli-v2 (frozen)
- **training_domains**: train_channel
- **n_train_chunks**: 6528
- **selected_layer**: 5
- **segmentation**: silero VAD + 4s chunks, RMS -26 dB
- **classifier**: MLP 768-256-2
- **hyperparameters**: lr=0.001, wd=0.0001, batch=64, dropout=0.3, seed=42, best epoch 1
- **device**: cuda
- **runtime_s**: 2.4000
- **val_original**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0004 (n=71)
- **val_channel_seen**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0038 (n=71)
- **val_channel_unseen**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0061 (n=71)
- **notes**: control; channel_unseen never used for selection

## 2026-09-12 08:06:45 - Robust Acoustic V2 - mixed

- **model**: facebook/wav2vec2-base-es-voxpopuli-v2 (frozen)
- **training_domains**: train_original, train_channel
- **n_train_chunks**: 10389
- **selected_layer**: 5
- **segmentation**: silero VAD + 4s chunks, RMS -26 dB
- **classifier**: MLP 768-256-2
- **hyperparameters**: lr=0.001, wd=0.0001, batch=64, dropout=0.3, seed=42, best epoch 1
- **device**: cuda
- **runtime_s**: 3.6000
- **val_original**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0001 (n=71)
- **val_channel_seen**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0024 (n=71)
- **val_channel_unseen**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0024 (n=71)
- **notes**: control; channel_unseen never used for selection

## 2026-09-12 08:06:45 - Robust Acoustic V2 - balanced

- **model**: facebook/wav2vec2-base-es-voxpopuli-v2 (frozen)
- **training_domains**: train_original, train_channel
- **n_train_chunks**: 10389
- **selected_layer**: 5
- **segmentation**: silero VAD + 4s chunks, RMS -26 dB
- **classifier**: MLP 768-256-2
- **hyperparameters**: lr=0.001, wd=0.0001, batch=64, dropout=0.3, seed=42, best epoch 1
- **device**: cuda
- **runtime_s**: 3.7000
- **val_original**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0000 (n=71)
- **val_channel_seen**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0016 (n=71)
- **val_channel_unseen**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0030 (n=71)
- **notes**: DEPLOYED as Model B; channel_unseen never used for selection

## 2026-09-12 08:07:33 - evaluation matrix - specialist vs robust v2

- **models**: specialist, specialist_silero, robust_v2
- **domains**: altur_val, channel_val_unseen
- **runtime_s**: 22.5000
- **worst_domain_accuracy**: {'specialist': np.float64(1.0), 'specialist_silero': np.float64(0.6667), 'robust_v2': np.float64(1.0)}
- **best_model**: specialist
- **notes**: channel_val_unseen never used for training or selection

## 2026-09-12 08:08:58 - Robust Acoustic V2 - orig_only

- **model**: facebook/wav2vec2-base-es-voxpopuli-v2 (frozen)
- **training_domains**: train_original
- **n_train_chunks**: 3861
- **selected_layer**: 5
- **segmentation**: silero VAD + 4s chunks, RMS -26 dB
- **classifier**: MLP 768-256-2
- **hyperparameters**: lr=0.001, wd=0.0001, batch=64, dropout=0.3, seed=42, best epoch 1
- **device**: cuda
- **runtime_s**: 3.1000
- **val_original**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0002 (n=71)
- **val_channel_seen**: acc=0.9859 bal_acc=0.9853 f1=0.9851 AUC=1.0000 EER=0.00% FAR=0.029 FRR=0.000 brier=0.0203 (n=71)
- **val_channel_unseen**: acc=0.9296 bal_acc=0.9265 f1=0.9206 AUC=1.0000 EER=0.00% FAR=0.147 FRR=0.000 brier=0.0757 (n=71)
- **notes**: control; channel_unseen never used for selection

## 2026-09-12 08:08:58 - Robust Acoustic V2 - channel_only

- **model**: facebook/wav2vec2-base-es-voxpopuli-v2 (frozen)
- **training_domains**: train_channel
- **n_train_chunks**: 6528
- **selected_layer**: 5
- **segmentation**: silero VAD + 4s chunks, RMS -26 dB
- **classifier**: MLP 768-256-2
- **hyperparameters**: lr=0.001, wd=0.0001, batch=64, dropout=0.3, seed=42, best epoch 1
- **device**: cuda
- **runtime_s**: 2.4000
- **val_original**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0004 (n=71)
- **val_channel_seen**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0038 (n=71)
- **val_channel_unseen**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0061 (n=71)
- **notes**: control; channel_unseen never used for selection

## 2026-09-12 08:08:58 - Robust Acoustic V2 - mixed

- **model**: facebook/wav2vec2-base-es-voxpopuli-v2 (frozen)
- **training_domains**: train_original, train_channel
- **n_train_chunks**: 10389
- **selected_layer**: 5
- **segmentation**: silero VAD + 4s chunks, RMS -26 dB
- **classifier**: MLP 768-256-2
- **hyperparameters**: lr=0.001, wd=0.0001, batch=64, dropout=0.3, seed=42, best epoch 1
- **device**: cuda
- **runtime_s**: 3.9000
- **val_original**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0001 (n=71)
- **val_channel_seen**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0024 (n=71)
- **val_channel_unseen**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0024 (n=71)
- **notes**: control; channel_unseen never used for selection

## 2026-09-12 08:08:58 - Robust Acoustic V2 - balanced

- **model**: facebook/wav2vec2-base-es-voxpopuli-v2 (frozen)
- **training_domains**: train_original, train_channel
- **n_train_chunks**: 10389
- **selected_layer**: 5
- **segmentation**: silero VAD + 4s chunks, RMS -26 dB
- **classifier**: MLP 768-256-2
- **hyperparameters**: lr=0.001, wd=0.0001, batch=64, dropout=0.3, seed=42, best epoch 1
- **device**: cuda
- **runtime_s**: 3.7000
- **val_original**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0000 (n=71)
- **val_channel_seen**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0016 (n=71)
- **val_channel_unseen**: acc=1.0000 bal_acc=1.0000 f1=1.0000 AUC=1.0000 EER=0.00% FAR=0.000 FRR=0.000 brier=0.0030 (n=71)
- **notes**: DEPLOYED as Model B; channel_unseen never used for selection

## 2026-09-12 08:36:12 - evaluation matrix - specialist vs robust v2

- **models**: specialist, specialist_silero, orig_only, robust_v2
- **domains**: altur_val, channel_val_seen, channel_val_unseen, external_ood, external_ood_channel, stress
- **runtime_s**: 1606.4000
- **worst_domain_accuracy**: {'specialist': np.float64(0.6204), 'specialist_silero': np.float64(0.6019), 'orig_only': np.float64(0.5556), 'robust_v2': np.float64(0.5463)}
- **best_model**: specialist
- **notes**: channel_val_unseen never used for training or selection

## 2026-09-12 08:38:23 - evaluation matrix - specialist vs robust v2

- **models**: specialist, specialist_silero, orig_only, robust_v2
- **domains**: altur_val, channel_val_seen, channel_val_unseen, external_ood, external_ood_channel, stress
- **runtime_s**: 1.0000
- **worst_domain_accuracy**: {'specialist': 0.6204, 'specialist_silero': 0.6019, 'orig_only': 0.5556, 'robust_v2': 0.5463}
- **worst_telephone_accuracy**: {'specialist': 0.6204, 'specialist_silero': 0.6481, 'orig_only': 0.7685, 'robust_v2': 0.8148}
- **best_model**: robust_v2
- **best_model_all_domains**: specialist
- **notes**: channel_val_unseen never used for training or selection; external_ood is clean non-telephone audio and is excluded from the deployment criterion but reported in full

## 2026-09-12 08:41:07 - evaluation matrix - specialist vs robust v2

- **models**: specialist, specialist_silero, orig_only, robust_v2
- **domains**: altur_val, channel_val_seen, channel_val_unseen, external_ood, external_ood_channel, stress
- **runtime_s**: 1.0000
- **worst_domain_accuracy**: {'specialist': 0.6204, 'specialist_silero': 0.6019, 'orig_only': 0.5556, 'robust_v2': 0.5463}
- **worst_telephone_accuracy**: {'specialist': 0.6204, 'specialist_silero': 0.6481, 'orig_only': 0.7685, 'robust_v2': 0.8148}
- **best_model**: robust_v2
- **best_model_all_domains**: specialist
- **notes**: channel_val_unseen never used for training or selection; external_ood is clean non-telephone audio and is excluded from the deployment criterion but reported in full

## 2026-09-12 08:42:04 - evaluation matrix - specialist vs robust v2

- **models**: specialist, specialist_silero, orig_only, robust_v2
- **domains**: altur_val, channel_val_seen, channel_val_unseen, external_ood, external_ood_channel, stress
- **runtime_s**: 0.8000
- **worst_domain_accuracy**: {'specialist': 0.6204, 'specialist_silero': 0.6019, 'orig_only': 0.5556, 'robust_v2': 0.5463}
- **worst_telephone_accuracy**: {'specialist': 0.6204, 'specialist_silero': 0.6481, 'orig_only': 0.7685, 'robust_v2': 0.8148}
- **best_model**: robust_v2
- **best_model_all_domains**: specialist
- **notes**: channel_val_unseen never used for training or selection; external_ood is clean non-telephone audio and is excluded from the deployment criterion but reported in full

