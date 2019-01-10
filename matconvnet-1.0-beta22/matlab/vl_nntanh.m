function out = vl_nntanh(x,dzdy)
%VL_NNTANH CNN sigmoid nonlinear unit.

y = tanh(x);

if nargin <= 1 || isempty(dzdy)
  out = y ;
else
  out = dzdy .* (1 - tanh(x).^2) ;
end
