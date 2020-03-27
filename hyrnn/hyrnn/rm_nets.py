import itertools
import torch.nn
import torch.nn.functional
import math
import geoopt.geoopt.manifolds.poincare.math as pmath
# import geoopt
from geoopt.geoopt.manifolds.poincare import PoincareBall
from geoopt.geoopt.tensor import ManifoldParameter

from torch.autograd import Function

class Distance(Function):
    @staticmethod
    def grad(x, v, sqnormx, sqnormv, sqdist, eps):
        alpha = (1 - sqnormx)
        beta = (1 - sqnormv)
        z = 1 + 2 * sqdist / (alpha * beta)
        a = ((sqnormv - 2 * torch.sum(x * v, dim=-1) + 1) / torch.pow(alpha, 2))\
            .unsqueeze(-1).expand_as(x)
        a = a * x - v / alpha.unsqueeze(-1).expand_as(v)
        z = torch.sqrt(torch.pow(z, 2) - 1)
        z = torch.clamp(z * beta, min=eps).unsqueeze(-1)
        return 4 * a / z.expand_as(x)

    @staticmethod
    def forward(ctx, u, v, eps):
        print(111111111111111111111111111111)
        squnorm = torch.clamp(torch.sum(u * u, dim=-1), 0, 1 - eps)
        sqvnorm = torch.clamp(torch.sum(v * v, dim=-1), 0, 1 - eps)
        sqdist = torch.sum(torch.pow(u - v, 2), dim=-1)
        ctx.eps = eps
        ctx.save_for_backward(u, v, squnorm, sqvnorm, sqdist)
        x = sqdist / ((1 - squnorm) * (1 - sqvnorm)) * 2 + 1
        # arcosh
        z = torch.sqrt(torch.pow(x, 2) - 1)
        return torch.log(x + z)

    @staticmethod
    def backward(ctx, g):
        u, v, squnorm, sqvnorm, sqdist = ctx.saved_tensors
        g = g.unsqueeze(-1)
        print( 1 )
        gu = Distance.grad(u, v, squnorm, sqvnorm, sqdist, ctx.eps)
        gv = Distance.grad(v, u, sqvnorm, squnorm, sqdist, ctx.eps)
        return g.expand_as(gu) * gu, g.expand_as(gv) * gv, None

# class RM_Linear(torch.nn.Module):
#     def __init__(self, input_features, output_features, bias=True):
#         super(RM_Linear, self).__init__()
#         self.input_features = input_features
#         self.output_features = output_features
#         # nn.Parameter is a special kind of Variable, that will get
#         # automatically registered as Module's parameter once it's assigned
#         # 这个很重要！ Parameters是默认需要梯度的！
#         self.weight = torch.nn.Parameter(torch.Tensor(output_features, input_features))
#         if bias:
#             self.bias = torch.nn.Parameter(torch.Tensor(output_features))
#         else:
#             # You should always register all possible parameters, but the
#             # optional ones can be None if you want.
#             self.register_parameter('bias', None)
#         # Not a very smart way to initialize weights
#         self.weight.data.uniform_(-0.1, 0.1)
#         if bias is not None:
#             self.bias.data.uniform_(-0.1, 0.1)
#     def forward(self, input):
#         # See the autograd section for explanation of what happens here.
#         return Distance.apply(input, self.weight, self.bias)
#         # 或者　return LinearFunction()(input, self.weight, self.bias)

# def rm_linear(input, weight, bias=None):
#     # First braces create a Function object. Any arguments given here
#     # will be passed to __init__. Second braces will invoke the __call__
#     # operator, that will then use forward() to compute the result and
#     # return it.
#     return RM_Linear()(input, weight, bias)#调用forward()

def mobius_linear(
    input,
    weight,
    bias=None,
    hyperbolic_input=True,
    hyperbolic_bias=True,
    nonlin=None,
    c=1.0,
):
    if hyperbolic_input:
        output = pmath.mobius_matvec(weight, input, c=c)
        print('x')
        print(weight.grad)
        print('x')
    else:
        output = torch.nn.functional.linear(input, weight)
        # output = rm_linear(input, weight)
        output = pmath.expmap0(output, c=c)
    if bias is not None:
        if not hyperbolic_bias:
            bias = pmath.expmap0(bias, c=c)
        output = pmath.mobius_add(output, bias, c=c)
    if nonlin is not None:
        output = pmath.mobius_fn_apply(nonlin, output, c=c)
    output = pmath.project(output, c=c)
    return output


def one_rnn_transform(W, h, U, x, b, c):
    W_otimes_h = pmath.mobius_matvec(W, h, c=c)
    U_otimes_x = pmath.mobius_matvec(U, x, c=c)
    Wh_plus_Ux = pmath.mobius_add(W_otimes_h, U_otimes_x, c=c)
    return pmath.mobius_add(Wh_plus_Ux, b, c=c)


def mobius_gru_cell(
    input: torch.Tensor,
    hx: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias: torch.Tensor,
    c: torch.Tensor,
    nonlin=None,
):
    W_ir, W_ih, W_iz = weight_ih.chunk(3)
    b_r, b_h, b_z = bias
    W_hr, W_hh, W_hz = weight_hh.chunk(3)

    z_t = pmath.logmap0(one_rnn_transform(W_hz, hx, W_iz, input, b_z, c), c=c).sigmoid()
    r_t = pmath.logmap0(one_rnn_transform(W_hr, hx, W_ir, input, b_r, c), c=c).sigmoid()

    rh_t = pmath.mobius_pointwise_mul(r_t, hx, c=c)
    h_tilde = one_rnn_transform(W_hh, rh_t, W_ih, input, b_h, c)

    if nonlin is not None:
        h_tilde = pmath.mobius_fn_apply(nonlin, h_tilde, c=c)
    delta_h = pmath.mobius_add(-hx, h_tilde, c=c)
    h_out = pmath.mobius_add(hx, pmath.mobius_pointwise_mul(z_t, delta_h, c=c), c=c)
    return h_out


def mobius_gru_loop(
    input: torch.Tensor,
    h0: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias: torch.Tensor,
    c: torch.Tensor,
    batch_sizes=None,
    hyperbolic_input: bool = False,
    hyperbolic_hidden_state0: bool = False,
    nonlin=None,
):
    if not hyperbolic_hidden_state0:
        hx = pmath.expmap0(h0, c=c)
    else:
        hx = h0
    if not hyperbolic_input:
        input = pmath.expmap0(input, c=c)
    outs = []
    if batch_sizes is None:
        input_unbinded = input.unbind(0)
        for t in range(input.size(0)):
            hx = mobius_gru_cell(
                input=input_unbinded[t],
                hx=hx,
                weight_ih=weight_ih,
                weight_hh=weight_hh,
                bias=bias,
                nonlin=nonlin,
                c=c,
            )
            outs.append(hx)
        outs = torch.stack(outs)
        h_last = hx
    else:
        h_last = []
        T = len(batch_sizes) - 1
        for i, t in enumerate(range(batch_sizes.size(0))):
            ix, input = input[: batch_sizes[t]], input[batch_sizes[t] :]
            hx = mobius_gru_cell(
                input=ix,
                hx=hx,
                weight_ih=weight_ih,
                weight_hh=weight_hh,
                bias=bias,
                nonlin=nonlin,
                c=c,
            )
            outs.append(hx)
            if t < T:
                hx, ht = hx[: batch_sizes[t+1]], hx[batch_sizes[t+1]:]
                h_last.append(ht)
            else:
                h_last.append(hx)
        h_last.reverse()
        h_last = torch.cat(h_last)
        outs = torch.cat(outs)
    return outs, h_last


class MobiusLinear(torch.nn.Linear):#(RM_Linear):
    def __init__(
        self,
        *args,
        hyperbolic_input=True,
        hyperbolic_bias=True,
        nonlin=None,
        c=1.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if self.bias is not None:
            if hyperbolic_bias:
                self.ball = manifold = PoincareBall(c=c)
                self.bias = ManifoldParameter(self.bias, manifold=manifold)
                with torch.no_grad():
                    self.bias.set_(pmath.expmap0(self.bias.normal_() / 4, c=c))
        with torch.no_grad():
            self.weight.normal_(std=1e-2)
        self.hyperbolic_bias = hyperbolic_bias
        self.hyperbolic_input = hyperbolic_input
        self.nonlin = nonlin

    def forward(self, input):
        return mobius_linear(
            input,
            weight=self.weight,
            bias=self.bias,
            hyperbolic_input=self.hyperbolic_input,
            nonlin=self.nonlin,
            hyperbolic_bias=self.hyperbolic_bias,
            c=self.ball.c,
        )

    def extra_repr(self):
        info = super().extra_repr()
        info += "c={}, hyperbolic_input={}".format(self.ball.c, self.hyperbolic_input)
        if self.bias is not None:
            info = ", hyperbolic_bias={}".format(self.hyperbolic_bias)
        return info


class MobiusDist2Hyperplane(torch.nn.Module):
    def __init__(self, in_features, out_features, c=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ball = ball = PoincareBall(c=c)
        self.sphere = sphere = geoopt.manifolds.Sphere()
        self.scale = torch.nn.Parameter(torch.zeros(out_features))
        point = torch.randn(out_features, in_features) / 4
        point = pmath.expmap0(point, c=c)
        tangent = torch.randn(out_features, in_features)
        self.point = ManifoldParameter(point, manifold=ball)
        with torch.no_grad():
            self.tangent = ManifoldParameter(tangent, manifold=sphere).proj_()

    def forward(self, input):
        input = input.unsqueeze(-2)
        distance = pmath.dist2plane(
            x=input, p=self.point, a=self.tangent, c=self.ball.c, signed=True
        )
        return distance * self.scale.exp()

    def extra_repr(self):
        return (
            "in_features={in_features}, out_features={out_features}, "
            "c={ball.c}".format(
                **self.__dict__
            )
        )


class MobiusGRU(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        nonlin=None,
        hyperbolic_input=True,
        hyperbolic_hidden_state0=True,
        c=1.0,
    ):
        super().__init__()
        self.ball = PoincareBall(c=c)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.weight_ih = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.Tensor(3 * hidden_size, input_size if i == 0 else hidden_size)
                )
                for i in range(num_layers)
            ]
        )
        self.weight_hh = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
                for _ in range(num_layers)
            ]
        )
        if bias:
            biases = []
            for i in range(num_layers):
                bias = torch.randn(3, hidden_size) * 1e-5
                bias = ManifoldParameter(
                    pmath.expmap0(bias, c=self.ball.c), manifold=self.ball
                )
                biases.append(bias)
            self.bias = torch.nn.ParameterList(biases)
        else:
            self.register_buffer("bias", None)
        self.nonlin = nonlin
        self.hyperbolic_input = hyperbolic_input
        self.hyperbolic_hidden_state0 = hyperbolic_hidden_state0
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in itertools.chain.from_iterable([self.weight_ih, self.weight_hh]):
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input: torch.Tensor, h0=None):
        # input shape: seq_len, batch, input_size
        # hx shape: batch, hidden_size
        is_packed = isinstance(input, torch.nn.utils.rnn.PackedSequence)
        if is_packed:
            input, batch_sizes = input[:2]
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = input.size(1)
        if h0 is None:
            h0 = input.new_zeros(
                self.num_layers, max_batch_size, self.hidden_size, requires_grad=False
            )
        h0 = h0.unbind(0)
        if self.bias is not None:
            biases = self.bias
        else:
            biases = (None,) * self.num_layers
        outputs = []
        last_states = []
        out = input
        for i in range(self.num_layers):
            out, h_last = mobius_gru_loop(
                input=out,
                h0=h0[i],
                weight_ih=self.weight_ih[i],
                weight_hh=self.weight_hh[i],
                bias=biases[i],
                c=self.ball.c,
                hyperbolic_hidden_state0=self.hyperbolic_hidden_state0 or i > 0,
                hyperbolic_input=self.hyperbolic_input or i > 0,
                nonlin=self.nonlin,
                batch_sizes=batch_sizes,
            )
            outputs.append(out)
            last_states.append(h_last)
        if is_packed:
            out = torch.nn.utils.rnn.PackedSequence(out, batch_sizes)
        ht = torch.stack(last_states)
        # default api assumes
        # out: (seq_len, batch, num_directions * hidden_size)
        # ht: (num_layers * num_directions, batch, hidden_size)
        # if packed:
        # out: (sum(seq_len), num_directions * hidden_size)
        # ht: (num_layers * num_directions, batch, hidden_size)
        return out, ht

    def extra_repr(self):
        return (
            "{input_size}, {hidden_size}, {num_layers}, bias={bias}, "
            "hyperbolic_input={hyperbolic_input}, "
            "hyperbolic_hidden_state0={hyperbolic_hidden_state0}, "
            "c={self.ball.c}"
        ).format(**self.__dict__, self=self, bias=self.bias is not None)
