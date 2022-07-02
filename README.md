# anomaly_detector
良品/不良品の2値分類

# 説明
画像を入力して、「良品」、「不良品」かを出力する

# インストール

1. サブモジュールの取得

```
git submodule update --init
```

2. PyTorch のインストール

GPU対応のPytouchを使用しているCUDAのversionに併せてインストールする。

```
# CUDA11.6 の例
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
```

3. 各種ライブラリのインストール

```
pip install -r requirements.txt
```

4. モデルの作成

```
$ python make_model.py
```

```
# 実行例
$ python make_model.py

█│ Running patchcore on metal_nut dataset.
 ╰─────────────────────────────────────────

   Could not find 'metal_nut' in 'ind_knn_ad/datasets/'. Downloading ...
100% [......................................................................] 165414484 / 165414484
   Found 'metal_nut' in 'datasets/'

   Training ...
   100%|██████████| 220/220 [00:09<00:00, 22.97it/s]
   Fitting random projections. Start dim = torch.Size([172480, 1536]).
   DONE.                 Transformed dim = torch.Size([172480, 297]).
   100%|██████████| 17247/17247 [00:17<00:00, 1002.41it/s]
   Testing ...
   100%|██████████| 115/115 [00:20<00:00,  5.56it/s]

   ╭────────────────────────┬────────────────────┬────────────────────╮
   │ Test results metal_nut │ image_rocauc: 0.98 │ pixel_rocauc: 0.96 │
   ╰────────────────────────┴────────────────────┴────────────────────╯
Model file creation is complete.
```

# 実行方法

```
usage: anomaly_detector.py [-h] [-l {warning,debug,info}] imagefile

Anormaly detector of metal_nut

positional arguments:
  imagefile

optional arguments:
  -h, --help            show this help message and exit
  -l {warning,debug,info}, --loglevel {warning,debug,info}
```

結果は、「ファイル名,スコア,判定」 の順に結果がでる

- スコア: 23 以上で不良と判定される
- 判定: 0が正常、1が不良


## 実行例1
```
$ python anomaly_detector.py ind_knn_ad/datasets/metal_nut/test/bent/000.png
ind_knn_ad/datasets/metal_nut/test/bent/000.png,28.2514705657959,1
```


## 実行例2
```
$ find ind_knn_ad/datasets/metal_nut/test -name "*.png" -exec ./anomaly_detector.py {} \;
ind_knn_ad/datasets/metal_nut/test/bent/009.png,33.15449142456055,1
ind_knn_ad/datasets/metal_nut/test/bent/001.png,30.217815399169922,1
ind_knn_ad/datasets/metal_nut/test/bent/006.png,25.360248565673828,1
ind_knn_ad/datasets/metal_nut/test/bent/023.png,27.282541275024414,1
ind_knn_ad/datasets/metal_nut/test/bent/010.png,33.41370391845703,1
ind_knn_ad/datasets/metal_nut/test/bent/015.png,29.72025489807129,1
ind_knn_ad/datasets/metal_nut/test/bent/012.png,32.17598342895508,1
ind_knn_ad/datasets/metal_nut/test/bent/007.png,22.04823875427246,0
ind_knn_ad/datasets/metal_nut/test/bent/000.png,28.2514705657959,1
ind_knn_ad/datasets/metal_nut/test/bent/019.png,30.045244216918945,1
ind_knn_ad/datasets/metal_nut/test/bent/013.png,28.608348846435547,1
ind_knn_ad/datasets/metal_nut/test/bent/005.png,29.480466842651367,1
ind_knn_ad/datasets/metal_nut/test/bent/024.png,31.660480499267578,1
ind_knn_ad/datasets/metal_nut/test/bent/002.png,23.833797454833984,1
ind_knn_ad/datasets/metal_nut/test/bent/004.png,28.59368133544922,1
ind_knn_ad/datasets/metal_nut/test/bent/008.png,30.47414779663086,1
ind_knn_ad/datasets/metal_nut/test/bent/011.png,33.60828399658203,1
ind_knn_ad/datasets/metal_nut/test/bent/003.png,30.33999252319336,1
ind_knn_ad/datasets/metal_nut/test/bent/018.png,29.775615692138672,1
ind_knn_ad/datasets/metal_nut/test/bent/017.png,31.539804458618164,1
ind_knn_ad/datasets/metal_nut/test/bent/016.png,28.38677406311035,1
ind_knn_ad/datasets/metal_nut/test/bent/020.png,32.017913818359375,1
ind_knn_ad/datasets/metal_nut/test/bent/022.png,27.260662078857422,1
ind_knn_ad/datasets/metal_nut/test/bent/014.png,26.003868103027344,1
ind_knn_ad/datasets/metal_nut/test/bent/021.png,34.96449279785156,1
ind_knn_ad/datasets/metal_nut/test/scratch/009.png,27.885541915893555,1
ind_knn_ad/datasets/metal_nut/test/scratch/001.png,27.165224075317383,1
ind_knn_ad/datasets/metal_nut/test/scratch/006.png,26.753475189208984,1
ind_knn_ad/datasets/metal_nut/test/scratch/010.png,26.703338623046875,1
ind_knn_ad/datasets/metal_nut/test/scratch/015.png,27.720273971557617,1
ind_knn_ad/datasets/metal_nut/test/scratch/012.png,25.93246841430664,1
ind_knn_ad/datasets/metal_nut/test/scratch/007.png,29.566198348999023,1
ind_knn_ad/datasets/metal_nut/test/scratch/000.png,22.604045867919922,0
ind_knn_ad/datasets/metal_nut/test/scratch/019.png,28.447063446044922,1
ind_knn_ad/datasets/metal_nut/test/scratch/013.png,29.39419937133789,1
ind_knn_ad/datasets/metal_nut/test/scratch/005.png,27.2574520111084,1
ind_knn_ad/datasets/metal_nut/test/scratch/002.png,26.31043243408203,1
ind_knn_ad/datasets/metal_nut/test/scratch/004.png,29.65691375732422,1
ind_knn_ad/datasets/metal_nut/test/scratch/008.png,31.062360763549805,1
ind_knn_ad/datasets/metal_nut/test/scratch/011.png,29.456525802612305,1
ind_knn_ad/datasets/metal_nut/test/scratch/003.png,26.149520874023438,1
ind_knn_ad/datasets/metal_nut/test/scratch/018.png,30.63979148864746,1
ind_knn_ad/datasets/metal_nut/test/scratch/017.png,25.875438690185547,1
ind_knn_ad/datasets/metal_nut/test/scratch/016.png,27.615982055664062,1
ind_knn_ad/datasets/metal_nut/test/scratch/020.png,24.695907592773438,1
ind_knn_ad/datasets/metal_nut/test/scratch/022.png,25.5764102935791,1
ind_knn_ad/datasets/metal_nut/test/scratch/014.png,26.392887115478516,1
ind_knn_ad/datasets/metal_nut/test/scratch/021.png,26.868186950683594,1
ind_knn_ad/datasets/metal_nut/test/good/009.png,19.41861915588379,0
ind_knn_ad/datasets/metal_nut/test/good/001.png,19.387836456298828,0
ind_knn_ad/datasets/metal_nut/test/good/006.png,20.361968994140625,0
ind_knn_ad/datasets/metal_nut/test/good/010.png,21.282316207885742,0
ind_knn_ad/datasets/metal_nut/test/good/015.png,21.268674850463867,0
ind_knn_ad/datasets/metal_nut/test/good/012.png,19.483665466308594,0
ind_knn_ad/datasets/metal_nut/test/good/007.png,23.833354949951172,1
ind_knn_ad/datasets/metal_nut/test/good/000.png,22.0274600982666,0
ind_knn_ad/datasets/metal_nut/test/good/019.png,22.26884651184082,0
ind_knn_ad/datasets/metal_nut/test/good/013.png,19.13372230529785,0
ind_knn_ad/datasets/metal_nut/test/good/005.png,21.56051254272461,0
ind_knn_ad/datasets/metal_nut/test/good/002.png,20.793853759765625,0
ind_knn_ad/datasets/metal_nut/test/good/004.png,20.45823097229004,0
ind_knn_ad/datasets/metal_nut/test/good/008.png,22.522531509399414,0
ind_knn_ad/datasets/metal_nut/test/good/011.png,22.86492347717285,0
ind_knn_ad/datasets/metal_nut/test/good/003.png,18.457605361938477,0
ind_knn_ad/datasets/metal_nut/test/good/018.png,17.86670684814453,0
ind_knn_ad/datasets/metal_nut/test/good/017.png,19.64804458618164,0
ind_knn_ad/datasets/metal_nut/test/good/016.png,21.761873245239258,0
ind_knn_ad/datasets/metal_nut/test/good/020.png,21.318851470947266,0
ind_knn_ad/datasets/metal_nut/test/good/014.png,19.246000289916992,0
ind_knn_ad/datasets/metal_nut/test/good/021.png,20.42943572998047,0
ind_knn_ad/datasets/metal_nut/test/color/009.png,30.653213500976562,1
ind_knn_ad/datasets/metal_nut/test/color/001.png,25.15968894958496,1
ind_knn_ad/datasets/metal_nut/test/color/006.png,27.404460906982422,1
ind_knn_ad/datasets/metal_nut/test/color/010.png,31.530174255371094,1
ind_knn_ad/datasets/metal_nut/test/color/015.png,24.082233428955078,1
ind_knn_ad/datasets/metal_nut/test/color/012.png,25.284284591674805,1
ind_knn_ad/datasets/metal_nut/test/color/007.png,26.763938903808594,1
ind_knn_ad/datasets/metal_nut/test/color/000.png,24.986366271972656,1
ind_knn_ad/datasets/metal_nut/test/color/019.png,21.757091522216797,0
ind_knn_ad/datasets/metal_nut/test/color/013.png,26.212602615356445,1
ind_knn_ad/datasets/metal_nut/test/color/005.png,21.91982078552246,0
ind_knn_ad/datasets/metal_nut/test/color/002.png,35.110694885253906,1
ind_knn_ad/datasets/metal_nut/test/color/004.png,19.067628860473633,0
ind_knn_ad/datasets/metal_nut/test/color/008.png,28.019437789916992,1
ind_knn_ad/datasets/metal_nut/test/color/011.png,30.516403198242188,1
ind_knn_ad/datasets/metal_nut/test/color/003.png,31.34888458251953,1
ind_knn_ad/datasets/metal_nut/test/color/018.png,21.817928314208984,0
ind_knn_ad/datasets/metal_nut/test/color/017.png,27.336767196655273,1
ind_knn_ad/datasets/metal_nut/test/color/016.png,24.652751922607422,1
ind_knn_ad/datasets/metal_nut/test/color/020.png,28.86632537841797,1
ind_knn_ad/datasets/metal_nut/test/color/014.png,29.42955207824707,1
ind_knn_ad/datasets/metal_nut/test/color/021.png,24.272314071655273,1
ind_knn_ad/datasets/metal_nut/test/flip/009.png,33.699527740478516,1
ind_knn_ad/datasets/metal_nut/test/flip/001.png,32.81901931762695,1
ind_knn_ad/datasets/metal_nut/test/flip/006.png,29.902305603027344,1
ind_knn_ad/datasets/metal_nut/test/flip/010.png,29.20659065246582,1
ind_knn_ad/datasets/metal_nut/test/flip/015.png,31.535247802734375,1
ind_knn_ad/datasets/metal_nut/test/flip/012.png,33.29957962036133,1
ind_knn_ad/datasets/metal_nut/test/flip/007.png,30.320878982543945,1
ind_knn_ad/datasets/metal_nut/test/flip/000.png,33.99467086791992,1
ind_knn_ad/datasets/metal_nut/test/flip/019.png,31.208545684814453,1
ind_knn_ad/datasets/metal_nut/test/flip/013.png,32.5325813293457,1
ind_knn_ad/datasets/metal_nut/test/flip/005.png,36.022125244140625,1
ind_knn_ad/datasets/metal_nut/test/flip/002.png,30.1946964263916,1
ind_knn_ad/datasets/metal_nut/test/flip/004.png,31.069286346435547,1
ind_knn_ad/datasets/metal_nut/test/flip/008.png,32.2264518737793,1
ind_knn_ad/datasets/metal_nut/test/flip/011.png,31.430532455444336,1
ind_knn_ad/datasets/metal_nut/test/flip/003.png,29.945419311523438,1
ind_knn_ad/datasets/metal_nut/test/flip/018.png,31.19550132751465,1
ind_knn_ad/datasets/metal_nut/test/flip/017.png,33.19103240966797,1
ind_knn_ad/datasets/metal_nut/test/flip/016.png,31.05829429626465,1
ind_knn_ad/datasets/metal_nut/test/flip/020.png,28.881484985351562,1
ind_knn_ad/datasets/metal_nut/test/flip/022.png,32.43429946899414,1
ind_knn_ad/datasets/metal_nut/test/flip/014.png,32.757225036621094,1
ind_knn_ad/datasets/metal_nut/test/flip/021.png,29.469892501831055,1
```

## 上記結果の混同行列

|            | 正常(予測) | 不良(予測) | 
| ---------- | ---------- | ---------- | 
| 正常(実際) | 21         | 1          | 
| 不良(実際) | 6          | 87         | 


# 使用したシステムの解説

SPADE, PaDiM, PatchCoreを使用できるライブラリ[Industrial KNN-based Anomaly Detection](https://github.com/rvorias/ind_knn_ad)を用いてPatchCoreの手法で正常な画像のみを教師データに用いて学習を行っている。

PatchCoreに関する詳細は[【2022年最新AI論文】画像異常検知AIの世界最先端手法「PatchCore」の論文を解説【CVPR 2022】](https://qiita.com/umapyoi/items/7c3e9b42388d576057b1)などが参考になる

他の異常検知の手法と比較すると下記の点で優位であると述べられている

- 深層学習モデル自体のトレーニングが不要
- 中間層の特徴量を利用することで、位置情報を保持
