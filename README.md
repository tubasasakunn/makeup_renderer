# Makeup Renderer
仮想化粧化粧を微分可能に実装したもの

# 使い方

## 学習（パラメータの最適化）
### 複数の場合
```shell
python train.py -c option/config.py -m multi -r res_multi_lip -i unit_test
```
`option/config.py`は設定ファイル．
`multi`は複数で在ることの指定．
` res_multi_lip`保存先のディレクトリ．
`unit_test`は訓練するデータのディレクトリ．
ただし構造は以下のようにファイル名は固定とします．
```
unit_test
└── canmake4
    ├── A_in.png
    ├── A_out.png
    ├── B_in.png
    ├── B_out.png

ただしA,Bはそれぞれ別の顔. _inは素顔画像で_outは化粧後画像．

```

### 単品の場合
```shell
python train.py -c option/config.py -m single -r res_multi_lip -i unit_test/canmake4/A_in.png -o unit_test/canmake4/A_out.png -t unit_test/canmake4/B_in.png
```

単品も複数も出力結果のファイルは以下のようなものにになります．

```shell
res_multi_lip/canmake4/
├── Eye
├── Face
├── Foundation
├── Lipstick
├── Makeup
├── all_img.png
├── in_img.png
├── out_img.png
├── params.pt
├── res_img.png
├── trans_in_img.png
├── trans_out_img.png
└── trans_res_img.png
```
各化粧工程の結果とパラメータが保存されます

## テスト（パラメータを使用し他の顔画像に適用）

### 画像に適用
```shell
python create.py -i test_img/img/image_0162.png -s res/res2.png -m test -p res_multi_lip/canmake4/params.pt
```
`res/res2.png`は保存ファイル名，`test_img.png`は適用画像，`res_multi_lip/canmake4/params.pt`は適用するパラメータのファイル


### ストローク動画の生成
徐々に化粧される様子を動画にするやつ
```shell
python create.py -i test_img/img/image_0162.png -s res/res2.mp4 -m test_stroke -p res_multi_lip/canmake4/params.pt
```

### 動画に適用
```shell
python create.py -i test_img/wakaiki2.mp4 -s res/res2.mp4 -m MakeVideo -p res_multi_lip/canmake4/params.pt
```