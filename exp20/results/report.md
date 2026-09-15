# Exp20 paired power sweep

Five paired seeds; descriptive evidence only. Bootstrap intervals from 20,000 resamples of seed differences. Five seeds do not justify strong significance claims.

## Mean performance and clipping geometry

Best final accuracy: [0.375]. Best accuracy AUC: [0.5].

Lowest clip fraction: [0.25]; lowest norm p99: [0.0]. These are separate geometry criteria, not a single utility optimum.

p,final_accuracy_mean,final_accuracy_sample_std,best_accuracy_mean,best_accuracy_sample_std,accuracy_auc_mean,accuracy_auc_sample_std,clip_fraction_mean,clip_fraction_sample_std,norm_p99_mean,norm_p99_sample_std,scale_match_mean,scale_match_sample_std,transformed_condition_proxy_mean,transformed_condition_proxy_sample_std
0.0,0.9255800000000001,0.00463055072318616,0.92702,0.0026948098263142703,3.6528,0.009400332440929967,0.30395165622234344,0.007759162767394122,305.68855163574216,17.938292835513472,1.1639208923827584,0.08946519081856362,214.94552553479076,45.97174583091485
0.125,0.9502,0.0031224989991991826,0.9503999999999999,0.0028844410203711715,3.73793,0.009522840962654173,0.26635817527770994,0.009414092490282541,474.550029296875,16.583458141342213,1.9380169634183049,0.06336988032695562,75.36412680125187,20.701731268559044
0.25,0.96052,0.002284075305238434,0.96052,0.002284075305238434,3.78031,0.010289521368849046,0.25428552329540255,0.009054257350823356,628.4282910156251,8.347656640809443,2.228750054976363,0.016456613644355387,32.026651474323224,11.5099712219848
0.375,0.96154,0.0023394443784796872,0.96154,0.0023394443784796872,3.79204,0.010500142856170935,0.25909521877765657,0.008769874154308303,698.8217163085938,20.378315487110935,1.6651886595159702,0.0031534904732616292,15.884157759898923,6.6944780717711705
0.5,0.9608000000000001,0.0019467922333931454,0.9608000000000001,0.0019467922333931454,3.7933399999999997,0.009536993761138922,0.27142427891492843,0.009339123765301525,744.9545812988281,29.658575247038026,1.0,0.0,8.485549622986575,3.850388610286635
0.625,0.95924,0.002021880312976008,0.95924,0.002021880312976008,3.7892400000000004,0.012240578009228207,0.2882805812358856,0.01069604639027465,799.0859423828125,32.15741425117512,0.5408151356796963,0.0032371853432628275,5.737630287169509,2.3342385108696004
0.75,0.9574400000000001,0.002265612499965563,0.9574400000000001,0.002265612499965563,3.7842700000000002,0.013920784819829694,0.30861979246139526,0.011378557303524743,870.8158862304688,44.71699687771717,0.2751838961285503,0.003624174011907053,6.159337078499567,1.7746863095612697
1.0,0.9528800000000001,0.003812086043100292,0.9528800000000001,0.003812086043100292,3.7668500000000003,0.019404799664000497,0.3529780977964401,0.012601108603421729,1059.0889135742186,142.70964017421377,0.06465761706671078,0.0017207950195216619,13.878399521623987,1.7608788904133343


## p=.375 − p=.5

Individual seed deltas, paired means, sample standard deviations and bootstrap 95% intervals:

p,reference_p,metric,paired_mean_delta,sample_std,ci95_low,ci95_high,delta_seed_42,delta_seed_7,delta_seed_123,delta_seed_2024,delta_seed_3407
0.375,0.5,final_accuracy,0.0007399999999999851,0.0006308724118235134,0.00021999999999995358,0.0011999999999999789,0.0008000000000000229,0.0013999999999999568,0.0011999999999999789,0.000500000000000056,-0.000200000000000089
0.375,0.5,accuracy_auc,-0.0012999999999998124,0.0038294908277732333,-0.003789999999999605,0.0021300000000000984,-0.004049999999999887,-0.0004500000000002835,0.005050000000000221,-0.003249999999999975,-0.0037999999999991374
0.375,0.5,clip_fraction,-0.01232906013727188,0.0022527680049776715,-0.014149300754070277,-0.010703793764114377,-0.012313038110733032,-0.0106002926826477,-0.01570845693349837,-0.010002672672271729,-0.013020840287208568
0.375,0.5,norm_p99,-46.13286499023436,10.403662439669185,-54.16424438476561,-37.32165405273436,-43.930816650390625,-46.2624450683594,-59.43211059570308,-30.819250488281227,-50.21970214843748


Raw paired results:

p,seed,final_accuracy,best_accuracy,accuracy_auc,clip_fraction,norm_p99,scale_match,transformed_condition_proxy
0.375,7,0.9638,0.9638,3.8018499999999995,0.2484174609184265,705.8174865722656,1.6665798651865777,14.884186120424307
0.375,42,0.9607,0.9607,3.78225,0.27087006270885466,685.258740234375,1.6645502032497919,10.328693632219004
0.375,123,0.9615,0.9615,3.79515,0.254403717815876,711.0195495605469,1.668603261686238,22.289610539591838
0.375,2024,0.9636,0.9636,3.8013,0.257251600921154,670.8710815429688,1.6601816285087339,23.251609363183597
0.375,3407,0.9581,0.9581,3.77965,0.26453325152397156,721.1417236328125,1.6660283389485102,8.666689144075876
0.5,7,0.9624,0.9624,3.8023,0.2590177536010742,752.079931640625,1.0,7.952291517758172
0.5,42,0.9599,0.9599,3.7862999999999998,0.2831831008195877,729.1895568847656,1.0,5.24711120353093
0.5,123,0.9603,0.9603,3.7901,0.2701121747493744,770.45166015625,1.0,12.148439544744075
0.5,2024,0.9631,0.9631,3.80455,0.26725427359342574,701.69033203125,1.0,12.729668916027869
0.5,3407,0.9583,0.9583,3.7834499999999993,0.2775540918111801,771.36142578125,1.0,4.350236932871831


final_accuracy stability (smaller sample std): p=0.375: 0.00233944, p=0.5: 0.00194679

More stable final_accuracy by sample std: [0.5] (ties listed).

p=.375 improves final_accuracy in 4/5 seeds; mean delta +0.00074, CI [+0.00022, +0.0012]. Not consistently positive across observed seeds.

accuracy_auc stability (smaller sample std): p=0.375: 0.0105001, p=0.5: 0.00953699

More stable accuracy_auc by sample std: [0.5] (ties listed).

p=.375 improves accuracy_auc in 1/5 seeds; mean delta -0.0013, CI [-0.00379, +0.00213]. Not consistently positive across observed seeds.

## Activation anisotropic geometry: p>0 versus p=0

Scale matching controls synthetic global RMS, not all private norms or optimization effects. p=0 is scalar identity, not ordinary DP-SGD.

p=0.125: final accuracy delta mean +0.02462, CI [+0.0218, +0.02744]; positive in 5/5 seeds. Consistent observed support for geometry utility.

p=0.25: final accuracy delta mean +0.03494, CI [+0.03132, +0.03856]; positive in 5/5 seeds. Consistent observed support for geometry utility.

p=0.375: final accuracy delta mean +0.03596, CI [+0.032, +0.03992]; positive in 5/5 seeds. Consistent observed support for geometry utility.

p=0.5: final accuracy delta mean +0.03522, CI [+0.03132, +0.03912]; positive in 5/5 seeds. Consistent observed support for geometry utility.

p=0.625: final accuracy delta mean +0.03366, CI [+0.02934, +0.03798]; positive in 5/5 seeds. Consistent observed support for geometry utility.

p=0.75: final accuracy delta mean +0.03186, CI [+0.02728, +0.03644]; positive in 5/5 seeds. Consistent observed support for geometry utility.

p=1.0: final accuracy delta mean +0.0273, CI [+0.02186, +0.03274]; positive in 5/5 seeds. Consistent observed support for geometry utility.

## Above p=.5: norm tail ↑, clipping ↑, accuracy ↓

p=0.625: joint pattern in 5/5 seeds; mean deltas {'final_accuracy': -0.0015600000000000058, 'accuracy_auc': -0.0040999999999997705, 'clip_fraction': 0.016856302320957174, 'norm_p99': 54.131361083984395}. Consistent observed pattern. This association does not establish causation.

p=0.75: joint pattern in 5/5 seeds; mean deltas {'final_accuracy': -0.0033600000000000296, 'accuracy_auc': -0.009069999999999644, 'clip_fraction': 0.03719551354646683, 'norm_p99': 125.86130493164062}. Consistent observed pattern. This association does not establish causation.

p=1.0: joint pattern in 5/5 seeds; mean deltas {'final_accuracy': -0.007920000000000016, 'accuracy_auc': -0.026489999999999726, 'clip_fraction': 0.0815538188815117, 'norm_p99': 314.13433227539065}. Consistent observed pattern. This association does not establish causation.

## Moderate optimum and under/over-conditioning

Both mean-utility maxima lie in p≈.375–.5: True. This is a candidate optimum region within this sweep.

Using mean-final-accuracy winner p=0.375 as the candidate sweet spot: improvement over p=0 in 5/5 seeds; higher tail/clipping and lower accuracy at p=.625,.75,1 in [5, 5, 5] seeds respectively. Consistent full pattern across observed seeds: True. This selection is descriptive and made after observing the sweep.

The p=0 comparisons and above-.5 joint counts quantify the proposed under-conditioning → sweet spot → over-preconditioning pattern; mixed seed signs limit its stability. Mean maxima alone do not establish a universal optimum.

Timing: algorithm = builder + private training; diagnostic CPU transfers and evaluation excluded; CUDA internal breakdown uses deferred Events. Runtime is secondary; small single-machine fluctuations do not establish power-specific speedups.

Biases use augmented activations. Spectral quantiles weight each eigenvalue once; RMS moments additionally weight layer output dimension. RDP uses the unchanged shuffled fixed-batch convention; clipping diagnostics are unnoised research measurements.
