# MPM トポロジー最適化への最大厚み制約追加

## Problem Statement

`examples/1_validation.ipynb` の材料点密度ベース MPM トポロジー最適化に、`examples/thickLSTO.ipynb` の PDE 型最大厚み制約を追加し、体積制約だけでは抑えられない厚すぎる構造特徴を制御できるようにする。

## Recommended Direction

推奨は、厚み制約を MMA の第 2 制約として追加する方針である。

`1_validation.ipynb` は、材料点密度 `x` を設計変数とし、密度フィルタ、しきい値投影、SIMP 的な材料パラメータ補間を通してコンプライアンスを最小化している。一方、`thickLSTO.ipynb` の厚み制約は、節点レベルセット `phi` から材料領域 `chi` を作り、幾何特徴 PDE、厚み違反評価、随伴方程式、Helmholtz 拡張を使って形状感度を得る構成である。

そのため、単純なコピーではなく、次の写像を明示的に作る必要がある。

```text
material-point density x
  -> filtered/projected material-point density rho_bar
  -> density-based level-set value phi_rho
  -> nodal material characteristic chi_node
  -> maximum-thickness PDE constraint G_thick
```

MMA は複数制約に対応しているため、最適化器側の拡張は小さい。難所は、`thickLSTO` の節点形状感度を `1_validation` の材料点密度設計変数に戻すチェーンルールである。

## Candidate Directions

### 1. MMA の第 2 制約として追加する

既存の体積制約 `G_vol <= 0` に加えて、厚み制約 `G_thick <= 0` を追加する。

実装上は `num_cons = 2` に変更し、`constr = [G_vol, G_thick]`、`grad_cons = [dG_vol/dx, dG_thick/dx]` を `_mma.update_mma(...)` に渡す。

この方向が最も自然で、既存の密度法と MMA の構造を保てる。

### 2. 厚み違反を目的関数ペナルティに足す

`objective_total = compliance + lambda * relu(G_thick)^2` のようにする。

実装は簡単だが、制約充足の制御は弱くなる。最初の挙動確認には使えるが、最終形としては弱い。

### 3. `1_validation` をレベルセット法へ寄せる

`thickLSTO` の `phi` 更新を主系にして、`1_validation` の密度法を置き換える。

これは大改造であり、MPM、JAX 勾配、密度フィルタ、材料点構造を作り替えることになる。今回の目的には過剰である。

## Key Assumptions to Validate

- 密度 `0.5` を境界とする疑似レベルセット表現から、厚み PDE が意味のある厚み違反場を返す。
- 材料点の疑似レベルセット値を正規化 GIMP shape function 投影で節点へ移しても、境界形状が十分安定する。
- 離散随伴法で求めた節点厚み制約勾配を、正規化 GIMP 投影の転置で材料点設計変数へ戻すことで、MMA の更新方向として機能する。
- 無次元パラメータ `h0`, `diffusion`, `ramp_epsilon` の組み合わせが、意図した最大厚みを表現する。
- 体積制約と厚み制約が強く競合しすぎず、MMA が実用的な設計更新を出せる。

## Confirmed Design Decisions

### 無次元厚みパラメータ

`h0` は mm や mesh cell 数ではなく、無次元の厚み上限パラメータとして扱う。厚み PDE には物理座標を直接渡さず、代表長さを使って無次元化した座標を渡す。

```text
L_ref = 0.09
x_hat = x / L_ref
```

`1_validation.ipynb` の解析領域は物理座標で `0.18 x 0.09` なので、無次元領域は `2 x 1` となり、`thickLSTO.ipynb` の領域と一致する。初期実装では `h0 = 0.1` を使用する。

### 密度から材料領域への変換

厚み制約には SIMP 後の剛性係数 `rho_E` ではなく、投影済み密度 `rho_bar` を使用する。密度 `0.5` をレベルセットのゼロ境界とみなし、材料点上で次を定義する。

```text
phi_rho = 2 * (rho_bar - 0.5)
```

- `rho_bar < 0.5`: `phi_rho < 0` なので空洞側
- `rho_bar = 0.5`: `phi_rho = 0` なので構造境界
- `rho_bar > 0.5`: `phi_rho > 0` なので構造側

完全な二値化を行うと境界以外の勾配が消えて最適化しにくくなるため、`thickLSTO.ipynb` と同じ C2 平滑化 Heaviside 関数をそのまま使用する。

$$
\chi(\phi)=
\begin{cases}
0 & \phi \leq -w,\\
\displaystyle
\frac{1}{2}
+x\left(
\frac{15}{16}
-\frac{5}{8}x^2
+\frac{3}{16}x^4
\right)
& -w<\phi<w,\\
1 & \phi \geq w,
\end{cases}
$$

ここで、

$$
x=\frac{\phi}{w}
$$

である。密度側では、遷移半幅を $\delta_\rho=0.05$ とする。このとき $\phi_\rho=2(\bar{\rho}-0.5)$ なので、平滑化幅は $w=2\delta_\rho=0.1$ となる。密度で表せば次の式と等価である。

$$
\chi(\bar{\rho})=
\begin{cases}
0 & \bar{\rho}\leq0.5-\delta_\rho,\\
\displaystyle
\frac{1}{2}
+x\left(
\frac{15}{16}
-\frac{5}{8}x^2
+\frac{3}{16}x^4
\right)
& |\bar{\rho}-0.5|<\delta_\rho,\\
1 & \bar{\rho}\geq0.5+\delta_\rho,
\end{cases}
$$

$$
x=\frac{\bar{\rho}-0.5}{\delta_\rho}
$$

したがって、$\bar{\rho}\leq0.45$ を空洞、$\bar{\rho}\geq0.55$ を構造とし、$0.45<\bar{\rho}<0.55$ の範囲だけを滑らかにつなぐ。構造境界の基準は引き続き $\bar{\rho}=0.5$ とする。

### 節点 `chi_node` の構築

`rho_bar` は材料点上の値だが、最大厚み PDE は格子節点上の場を未知変数として解く。そのため、材料点上の `phi_rho` を格子節点へ投影する処理が必要になる。

推奨する変換は、MPM と整合する正規化 GIMP shape function 投影である。

```text
phi_node[i]
  = sum_p(N_ip * V_p * phi_rho[p])
    / sum_p(N_ip * V_p)

chi_node[i] = smooth_characteristic(phi_node[i])
```

ここで `N_ip` は材料点 `p` に対する格子節点 `i` の GIMP shape function、`V_p` は材料点体積である。要素平均を二段階で節点へ移す方法より、既存の MPM の粒子・格子対応を直接利用でき、境界情報を失いにくい。

投影分母 `sum_p(N_ip * V_p)` がゼロの節点は、材料点の影響がない節点として次のように空洞へ固定する。

```text
phi_node = -1
chi_node = 0
projection gradient = 0
```

平滑化 Heaviside は GIMP 投影後の節点値に適用する。

```text
chi_node[i] = smooth_characteristic(phi_node[i], w=0.1)
```

これは、投影後の等価節点密度 $\bar{\rho}_{node}=(\phi_{node}+1)/2$ に対して、上記の $\delta_\rho=0.05$ の式を適用することと等価である。

投影に使う材料点体積は、厚み制約パラメータで選択可能にする。

```python
projection_volume_mode: Literal["reference", "current"] = "reference"
```

- `reference`: `volume0` を使い、変形前の参照配置における設計厚みを評価する。
- `current`: `volume` を使い、荷重後の現在配置における設計厚みを評価する。

既定値は `reference` とする。`reference` では `volume0`, 初期 `coord`, `domain_length0` から GIMP map を構築する。`current` では `volume`, 現在の `coord`, `domain_length` から GIMP map を再構築する。

`current` は変形状態にも依存するため、初期実装では変形依存性の感度を無視し、各 MMA 反復で得られた現在配置を厚み制約評価中の固定値として扱う実験的オプションとする。既定の最適化と感度検証には `reference` を使用する。

### 厚み制約の許容値

厚み違反積分の許容量として `j_max = 0.01` を使用する。これは数値ソルバの収束許容誤差ではなく、厚み制約の可行領域を定めるモデルパラメータである。

```text
G_thick = integral(chi_node * thickness_violation) - 0.01
```

MMA には `G_thick <= 0` の第 2 制約として渡す。

`1e-4` は MMA 内部の設定値にはせず、最適化終了後の可行性判定基準として使用する。

```text
G_vol <= 1e-4
G_thick <= 1e-4
```

`j_max` は厚み制約モデルの許容違反量、`1e-4` は得られた設計を制約充足と判断するための数値的な許容誤差として区別する。

### 最適化と検証

目的関数はコンプライアンスのままとし、MMA が体積制約とともに `G_thick <= 0` を満たす方向へ設計を更新する。`thickLSTO.ipynb` と同様に、厚み PDE の感度を設計更新へ反映する。

最大厚み場や違反場の可視化は追加の最適化基準ではなく、`G_thick` の低下が正しい場所の材料除去によって起きているかを確認する診断情報として使用する。

## MVP Scope

最小実装は、最初から最適化全体へ入れるのではなく、現在の `rho_phys` に対して厚み制約を評価できるところまでにする。

1. `moto/src/thickness_constraint.py` を作る。
2. `thickLSTO.ipynb` から以下をモジュール化して移植する。
   - `MaximumThicknessParams`
   - 幾何特徴 PDE state solve
   - 厚み違反評価
   - 随伴 solve
   - 離散随伴による節点制約勾配
   - `analyze_maximum_thickness`
3. `1_validation.ipynb` 側で、`rho_bar` から `phi_rho` を作り、正規化 GIMP shape function 投影で `phi_node` と `chi_node` を構築する関数を追加する。
4. GIMP 投影の体積重みを `projection_volume_mode` で `volume0` または `volume` から選択できるようにする。
5. 厚み PDE 用の節点座標を `L_ref = 0.09` で無次元化する。
6. `rho_phys` から `G_thick`、厚み場、違反場を可視化する。
7. 離散随伴勾配を有限差分と比較してから、MMA の第 2 制約へ接続する。
8. `max_iter=5` から `10` 程度で、`G_thick` が減少するか確認する。

## Implementation Sketch

既存の体積制約は JAX/JIT のまま維持し、SciPy 疎行列 solve を使う厚み制約は JIT の外で評価する。MMA ループ内で両者を結合する。

```python
g_vol, dg_vol = volume_constraint_jax(...)
g_thick, dg_thick = thickness_constraint_scipy(...)

constr = np.array([
    [g_vol],
    [g_thick],
])
grad_cons = np.vstack([
    np.asarray(dg_vol).reshape(1, -1),
    np.asarray(dg_thick).reshape(1, -1),
])
```

`constr` の形状は `(2, 1)`、`grad_cons` の形状は `(2, num_design_var)` とする。

厚み制約には、`thickLSTO.ipynb` の材料除去方向をそのまま MMA 勾配として流用しない。厚み PDE の離散残差を基準に、真の節点制約勾配 `dG_thick/dchi_node` を随伴法で導出する。

その後、勾配を次の経路で設計変数へ戻す。

```text
dG_thick/dx
  = dG_thick/dchi_node
  * dchi_node/dphi_node
  * dphi_node/dphi_rho
  * dphi_rho/drho_bar
  * drho_bar/drho_tilde
  * drho_tilde/dx
```

`dphi_node/dphi_rho` には正規化 GIMP 投影の転置を使う。`drho_tilde/dx` は既存の `density_filter`、`drho_bar/drho_tilde` は `threshold_filter` の微分を使う。

節点勾配から設計変数勾配への実装経路は次の通りである。

```text
node constraint gradient
  -> transpose of normalized GIMP projection
  -> material-point density gradient
  -> threshold-filter derivative
  -> transpose of density filter
  -> design-variable gradient
```

勾配の符号と大きさは、複数の設計変数を選んだ中心有限差分で検証する。有限差分と一致しない限り MMA の厚み制約には接続しない。

## Not Doing

- 最初から完全な JAX 微分可能 PDE にすることはしない。厚み PDE は SciPy solve と随伴感度でまず十分である。
- `1_validation.ipynb` 全体をレベルセット法へ置き換えない。今回の目的は密度法 MPM への厚み制約追加である。
- `thickLSTO.ipynb` のコードを notebook に大量に貼り付けない。再利用できるように `moto/src` 側へモジュール化する。
- 厳密な連続形状微分の完全移植を最初の MVP に含めない。まずは離散写像上の近似感度で制約が効くか検証する。

## Open Questions

現時点で実装を開始するための未決事項はない。$\delta_\rho=0.05$、`h0 = 0.1`、`j_max = 0.01` の妥当性は、実装後に有限差分感度、厚み場の可視化、制約履歴で検証する。
