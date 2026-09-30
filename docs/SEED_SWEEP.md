# Seed-robustness sweep

Same AOI layout and change sites; every seed redraws sensor noise, cloud and shadow fields,
building footprints and field textures (gain, mis-registration and haze stay fixed). Command: `python scripts/seed_sweep.py`.

| Seed | Events | Obj P | Obj R | Type acc | Earliest acc | Pixel P / R | FA px crop / haze / cloud | Naive FA px crop / haze / cloud | Macro P@10 |
|---|---|---|---|---|---|---|---|---|---|
| 11 | 5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 / 0.994 | 0 / 0 / 0 | 341,693 / 78,295 / 161,765 | 1.0 |
| 42 | 5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 / 0.998 | 0 / 0 / 0 | 334,694 / 77,716 / 162,663 | 1.0 |
| 7 | 5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 / 0.997 | 0 / 0 / 0 | 332,434 / 78,798 / 162,267 | 1.0 |
