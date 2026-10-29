| method | strong 224 speedup 8->70 SMs (eff.) | strong 896 speedup 8->70 SMs (eff.) | weak eff. 70 SMs | 70-SM ms 896 / cudnn |
|---|---|---|---|---|
| cudnn | 4.5x (51%) | 4.5x (51%) | 52% | 1.00x |
| naive | 8.4x (96%) | 8.9x (102%) | 98% | 3.09x |
| tiled | 8.0x (92%) | 9.0x (102%) | 98% | 2.53x |
| im2col | 7.8x (89%) | 8.7x (99%) | 95% | 1.85x |
| im2col_reg | 5.9x (68%) | 7.7x (89%) | 82% | 1.39x |
| implicit | 4.3x (50%) | 6.7x (76%) | 75% | 0.55x |
| gridstride | 8.0x (91%) | 8.2x (94%) | 87% | 4.13x |
