使い方

この仕組みは、10x10マスの簡易的な都市シミュレーターです。
Genesisで動作させて、学習させたモデルをindex.htmlで作成されたシミュレーターにて動作させることが可能です。

1.Genesisでの動作

genesis-mesa.pyをgenesisのsampleフォルダに入れる

$python genesis-mesa.pyで実行

実行後、policy.onnxをダウンロード

2.Google Colabで学習

city_sim_training_v4.ipynbをGoogleColabで読み込み

実行後、policy.onnxをダウンロード

3.シミュレーターで実行

index.htmlからpolicy.onnxを読み込み
