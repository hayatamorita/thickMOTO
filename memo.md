1. 厚み違反はハード判定ではない

  厚み制約では、

  $$
  q=1-\frac{h_0}{h_s}
  $$

  を平滑Ramp関数へ入れています。

  $$
  R(q)=\frac{1}{2}\left(q+\sqrt{q^2+\epsilon}\right)
  $$

  ramp_epsilon=0.001なので、$h_s<h_0$でも$R(q)$とその微分は厳密にはゼロになりません。したがっ
  て、$h_s<0.3$の領域も弱く感度へ寄与します。

  2. 厚み感度はPDEを介した非局所感度

  h_sはその場所の厚み表示ですが、感度は

  構造形状の変化
  → 厚みPDEの解が領域全体で変化
  → G_thickが変化

  を随伴法で計算しています。

  そのため、ある境界を動かすことで別の場所の厚み違反が改善する場合、その境界に感度が出ます。感度は単純
  な

  h_s > 0.3 の場所だけ非ゼロ

  というマスクではありません。

  3. 現在比較している図が同じ計算結果ではない

  統合図のh_sは、

  h_s = thickness_result.analysis.thickness

  ですが、current raw thickness sensitivityは、

  current_thickness_result
  current_grad_thickness

  をcurrent_beta=15で再計算したものです。

  したがって、比較するなら、

  current_h_s = current_thickness_result.analysis.thickness

  を見る必要があります。

  また、

  - current_h_s：オイラー格子節点上
  - current_grad_thickness：材料点上

  なので、座標も直接一対一では対応しません。

  現在のcurrent G_thick=4.392e-02は正なので、領域全体としては厚み制約に違反しています。ただし、感度が
  出た各材料点で必ず$current_h_s>0.3$という意味ではありません。

  正確に確認するには、同じcurrent_thickness_resultから以下を並べて表示する必要があります。

  current_h_s
  current_h_s > 0.3 の違反マスク
  current analysis.gradient_characteristic
  current_grad_thickness

  これにより、「実際の違反領域」と「違反改善に寄与する境界感度」を区別できます。コードは変更していませ
  ん。
----
• 2つ目のcharacteristic関数は、レベルセット場phi_nodeを「構造か空洞か」を表す連続的な材料指示関数chi_nodeへ変
  換しています。

  phi_node < 0  -> 空洞      -> chi_node = 0
  phi_node > 0  -> 構造      -> chi_node = 1
  phi_node ≈ 0  -> 境界領域  -> 0 < chi_node < 1

  phi_nodeは次のように作られています。

  $$
  \phi_{particle}=2(\rho_{bar}-0.5)
  $$

  これをGIMP投影してphi_nodeにします。その後、thickLSTO.ipynbと同じ平滑化ヘビサイド関数を適用します。

  $$
  \chi(\phi)=
  \begin{cases}
  0 & \phi \le -w \
  \frac{1}{2}
  +\frac{15}{16}\frac{\phi}{w}
  -\frac{5}{8}\left(\frac{\phi}{w}\right)^3
  +\frac{3}{16}\left(\frac{\phi}{w}\right)^5
  & |\phi|<w \
  1 & \phi \ge w
  \end{cases}
  $$

  ここでw=characteristic_width=0.025です。

  厚みPDEは、このchi_nodeを使って「どこが構造領域か」を判定します。

  その微分は、

  $$
  \frac{\partial\chi}{\partial\phi}

  \begin{cases}
  \frac{15}{16w}
  \left(1-\left(\frac{\phi}{w}\right)^2\right)^2
  & |\phi|<w \
  0 & |\phi|\ge w
  \end{cases}
  $$

  となります。

  したがって、厚みPDEから得られたanalysis.gradient_characteristicは、次の乗算によってphi_nodeに対する感度へ変
  換されます。

  gradient_phi_nodes = (
      analysis.gradient_characteristic
      * smooth_characteristic_derivative(phi_nodes, width=w)
  )

  つまり2つ目の処理は、単に滑らかにするだけではなく、厚みPDEの感度を構造境界phi_node=0付近だけに限定する処理
  です。|phi_node| >= 0.025の構造内部と空洞内部では、この段階で感度がゼロになります。