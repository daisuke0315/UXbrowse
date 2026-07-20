# 設計判断記録

現在の主要な設計判断と、その理由を簡潔に記録します。

## 採用している方針

- **ブラウザ内で処理する**
  - Three.js表示、DICOM／MHA読込、MPR生成をJavaScriptへ集約します。
  - 外部通信を減らせる一方、大規模データではCPU・メモリが制約になります。

- **DICOM患者座標を基準にする**
  - `Image Position Patient`、`Image Orientation Patient`、`Pixel Spacing`を使用します。
  - Three.jsの1 unitを1 mmとして扱い、既定の座標変換は恒等変換です。

- **voxel中心の範囲を使う**
  - Volume中心と物理範囲は`(dimensions - 1) * spacing`を基準にします。
  - MPRサンプル位置と3D空間の表示位置を一致させるためです。

- **MPR中心をカメラ前方に置く**
  - `target = camera.position + camera.forward * MPR距離`で定義します。
  - 通常の既定距離は600 mm、みかん・バナナのデモは300 mmです。
  - `OrbitControls.target`やモデル中心はMPR中心として使用しません。

- **固定断面でも表示位置は視線に追従させる**
  - Axial、Coronal、Sagittalでは断面方向だけを固定します。
  - 断面中心はカメラ前方の注視点へ追従し、CT中央へ固定しません。

- **MPRを3D空間内の実平面として表示する**
  - MPRの`center / u / v / n`と物理FOVをThree.jsへ反映します。
  - サブ画面と3D重畳面は同じ生成画像を共有します。

- **前景透過にはOITを使用する**
  - 対応GPUではWeighted Blended OITで半透明を合成します。
  - OIT非対応環境ではAlpha Hashへフォールバックします。
  - Alpha 0は手前側を切断し、Alpha 1は透過しません。

- **外観モデルはUnlit・両面表示を既定にする**
  - 撮影済みtextureの色を照明で変えず、面方向の不整合による欠落を抑えます。
  - 通常表示ではdepth test／depth writeを有効に保ちます。

- **Reverse-Zと動的near／farを使用する**
  - 対応GPUではReverse-Zを有効にします。
  - 未対応環境では通常depthへ戻し、near／farを表示範囲へ追従させます。

- **3D表現形式を固定しない**
  - 現在はGLB／OBJを使用します。
  - Gaussian Splattingを追加する場合も、座標・カメラ・MPR平面の契約は共通化します。

これらを変更する場合は、表示上の修正ではなく操作仕様または座標仕様の変更として扱います。
