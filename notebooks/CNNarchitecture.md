                         GALAXAI CUSTOM CNN V2
                                 │
                                 ▼
                     Input: 224 × 224 × 3
                                 │
                                 ▼
                    ┌──────────────────────┐
                    │   CONV BLOCK 1       │
                    │   3 → 32             │
                    │   2 × Conv 3×3       │
                    │   BatchNorm + ReLU   │
                    │   MaxPool             │
                    └──────────────────────┘
                                 │
                         32 × 112 × 112
                                 │
                                 ▼
                    ┌──────────────────────┐
                    │   CONV BLOCK 2       │
                    │   32 → 64             │
                    │   2 × Conv 3×3       │
                    │   BatchNorm + ReLU   │
                    │   MaxPool             │
                    └──────────────────────┘
                                 │
                          64 × 56 × 56
                                 │
                                 ▼
                    ┌──────────────────────┐
                    │   CONV BLOCK 3       │
                    │   64 → 128            │
                    │   2 × Conv 3×3       │
                    │   BatchNorm + ReLU   │
                    │   MaxPool             │
                    └──────────────────────┘
                                 │
                         128 × 28 × 28
                                 │
                                 ▼
                    ┌──────────────────────┐
                    │   CONV BLOCK 4       │
                    │   128 → 256           │
                    │   2 × Conv 3×3       │
                    │   BatchNorm + ReLU   │
                    │   MaxPool             │
                    └──────────────────────┘
                                 │
                         256 × 14 × 14
                                 │
                                 ▼
                    AdaptiveAvgPool2d(1,1)
                                 │
                           256 × 1 × 1
                                 │
                                 ▼
                              Flatten
                                 │
                            256 features
                                 │
                                 ▼
                         Linear 256 → 256
                                 │
                               ReLU
                                 │
                         Dropout = 0.40
                                 │
                    ┌────────────┴─────────────┐
                    │                          │
                    ▼                          ▼
              BINARY HEADS              MULTICLASS HEADS
                    │                          │
        ┌───────────┼───────────┐       ┌──────┴──────┐
        │           │           │       │             │
        ▼           ▼           ▼       ▼             ▼
     8 × 1       ...         8 × 1   Bulge        Roundedness
                                      256→4          256→3
        │                                │             │
        ▼                                ▼             ▼
    8 logits                         4 logits       3 logits

                         TOTAL = 15 LOGITS