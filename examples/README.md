

## hetero_conv_1
hetero_conv_1: 测试第一版，最简单的卷积模式
$\mathbf{h}_s^{(l)} \leftarrow q_t + \operatorname{SUM}\left(\sigma\left(\mathbf{W}^{(l)} \cdot\mathbf{q}_t^{(l-1)})\right.\right.)$
$\mathbf{h}_t^{(l)} \leftarrow q_s + \operatorname{SUM}\left(\sigma\left(\mathbf{W}^{(l)} \cdot\mathbf{q}_s^{(l-1)})\right.\right.)$

