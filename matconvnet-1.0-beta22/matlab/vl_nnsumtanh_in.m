function out = vl_nnsumtanh_in(x,dzdy)
%VL_NNTANH CNN sigmoid nonlinear unit.

% pa = [0.4923 3.2585 0.5425 0.8268]; input range 0-2
% pa = single([0.3282 0.3560 0.6628 2.2374]);
pa = single([0.6734    0.6021    0.3165    0.0851]);

y = pa(1)*tanh(pa(2)*x)+pa(3)*tanh(pa(4)*x);

if nargin <= 1 || isempty(dzdy)
  out = y ;
else
    
    out = dzdy .* (1 - tanh(x*pa(2)).^2)*pa(2)+(1-tanh(x*pa(4)).^2)*pa(4);
 
end
