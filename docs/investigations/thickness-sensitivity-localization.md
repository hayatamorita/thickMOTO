# 厚み感度の局所化・斑点化に関する調査

## 状態

- 調査日: 2026-06-14
- 対象: `examples/1_validation.ipynb`
- 目的: コンプライアンス感度に比べ、厚み感度が局所的かつ斑点状になる原因を切り分ける
- 本書では原因調査までを扱い、対策は未決定とする

## 再現条件

最適化条件は以下だった。

```python
max_iter=10
move_limit=1e-2
num_load_steps=1
constraint_mode="volume_and_thickness"
thickness_start_epoch=0
```

密度投影と厚み評価の主要パラメータは以下である。

```python
thresh_beta = min(32.0, 1.0 + 2.5 * mma_state.epoch)
eta = 0.5
characteristic_width = 0.025
density_filter_radius = 1.5 * mesh.elem_size[0]
projection_volume_mode = "reference"
```

最終反復は`epoch=9`なので、最適化中の密度投影係数は次になる。

$$
\beta = 1 + 2.5 \times 9 = 23.5
$$

## 観察結果

### MMA実効感度の大きさ

最終反復の平均絶対感度は以下だった。

```text
S_comp  = 2.520e-3
S_thick = 2.059e-3
cos     = -0.728
```

厚み制約を`epoch=0`から有効化した場合、コンプライアンスと厚みのMMA実効感度は同じ桁になった。
一方、感度方向は強く競合している。

### 空間分布

- コンプライアンス感度は構造内部に連続的に分布した
- 生の厚み感度`grad_thickness`は、既に斑点状だった
- MMA実効厚み感度は、生の厚み感度と同じ位置に分布した
- `lambda_thick`は全点共通のスカラーなので、斑点の発生位置は変えない
- 最適化後の設計を`current_beta=4`で再評価すると、厚み感度は広い領域へ滑らかに広がった

以上から、斑点化はMMA乗数より前の厚み感度計算で発生している。

## 厚み感度の計算経路

現在の設計感度は次の経路で計算される。

```text
design x
-> density_filter
-> tanh density projection
-> rho_bar
-> phi_particles = 2 (rho_bar - 0.5)
-> GIMP projection to Euler nodes
-> smooth characteristic
-> thickness PDE and adjoint
-> characteristic derivative
-> GIMP transpose projection
-> density projection derivative
-> density_filter transpose
-> grad_thickness
```

コード上では概ね以下に対応する。

```python
gradient_phi_nodes = (
  analysis.gradient_characteristic
  * smooth_characteristic_derivative(phi_nodes, width=characteristic_width)
)
gradient_phi_particles = projection.transpose(gradient_phi_nodes)
gradient_rho_tilde = 2.0 * gradient_phi_particles * threshold_derivative
gradient_design = density_filter.T @ gradient_rho_tilde
```

## 主原因

最も有力な原因は、次の2つの微分による二重の局在化である。

```text
tanh密度投影の微分
×
smooth characteristicの微分
```

### characteristicによる局在化

`characteristic_width=0.025`では、次の範囲だけで微分が非ゼロになる。

$$
|\phi| < 0.025
$$

また、$\phi=2(\rho_{bar}-0.5)$なので、対応する密度範囲は以下となる。

$$
0.4875 < \rho_{bar} < 0.5125
$$

### 密度投影による追加の局在化

最終反復の$\beta=23.5$で上記範囲を投影前密度$\tilde{\rho}$へ戻すと、概ね以下となる。

$$
0.49894 < \tilde{\rho} < 0.50106
$$

感度が有効な密度幅は約$0.0021$しかない。
この狭い範囲に入る材料点とオイラー節点が離散的になるため、感度が斑点状になる。

## 悪化要因

### GIMP投影の離散化

- 材料点は各要素方向に4点ある
- 厚みPDEはオイラー格子節点上で解く
- 狭い感度帯に入るオイラー節点が少ない
- GIMP転置後も、その節点に接続する材料点へ局所的に感度が戻る

### 密度フィルターの作用範囲

密度フィルター半径は`1.5 mesh cell`である。
斑点を局所的に広げる効果はあるが、構造全体へ感度を延長するものではない。

### Helmholtz感度拡張がない

`examples/thickLSTO.ipynb`では、境界上のshape derivativeをHelmholtz PDEで領域へ拡張する。
現在の`1_validation.ipynb`には、この感度拡張がない。

### 参照配置での評価

現在は`projection_volume_mode="reference"`である。
大変形後の構造を表示する場合、参照配置で計算した感度位置との見かけ上のずれが生じ得る。
ただし、今回の斑点化は生感度自体に存在するため、主原因ではない。

## 原因ではないもの

### MMA乗数

$$
S_{thick}^{MMA}
=
\lambda_{thick}\frac{\partial G_{thick}}{\partial x}
$$

$\lambda_{thick}$は全点共通のスカラーである。
感度の絶対値は変えるが、非ゼロ位置や斑点形状は変えない。

### 可視化のカラースケール

各感度図は個別の対称カラースケールを使用している。
図同士の色の濃さは直接比較できないが、生厚み感度そのものが斑点状であることは確認済みである。

## 現時点の結論

原因の優先順位は次の通りである。

1. $\beta=23.5$と`characteristic_width=0.025`による二重の局在化
2. 狭い感度帯に対するオイラー格子とGIMP投影の離散化
3. Helmholtz感度拡張がないこと
4. 密度フィルター半径が局所的であること

`current_beta=4`では感度が広がるため、$\beta$の影響は実験的にも確認できた。
ただし、$\beta=4$では構造内部まで感度が広がり過ぎるため、そのまま採用するとは限らない。

## 次に必要な診断

以下を同一反復で順番に可視化し、斑点が最初に現れる段階を確定する。

1. `analysis.gradient_characteristic`
2. `smooth_characteristic_derivative(phi_nodes)`
3. `gradient_phi_nodes`
4. `projection.transpose(...)`後の`gradient_phi_particles`
5. `threshold_derivative`
6. `gradient_rho_tilde`
7. `density_filter.T`後の`gradient_design`

対策立案では、厚み用$\beta$の分離、characteristic幅、格子解像度、Helmholtz拡張を比較対象とする。
