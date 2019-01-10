function [out, w1u] = vl_nnconv_lsuv(x, w1, w2, ...
    pad, ...
    stride, ...
    dilate, ...
    opts, ...
    cudnn)

% orthonormal pre-initialization as Saxe et. al 2014

[k1,k2,m,n] = size(w1);
mwi = reshape(w1,k1*k2*m,n);
[u,~,v]=svd(mwi,'econ');

if sum(size(mwi) == size(u))>1
    
    w1u = u;
    
elseif sum(size(mwi) == size(u'))>1
    
    w1u = u';
    
elseif sum(size(mwi) == size(v))>1
    
    w1u = v;
    
elseif sum(size(mwi) == size(v'))>1
    
    w1u = v';
    
end

w1u = reshape(w1u,k1,k2,m,n);

% Layer-sequential unit-variance Mishkin et. al 2015 Lsuv


x1 = vl_nnconv(x, w1u, w2, ...
    'pad', pad, ...
    'stride', stride, ...
    'dilate', dilate, ...
    opts{:}, ...
    cudnn{:}) ;
xs1 = vl_nnsumtanh_in(x1) ;
ti = 0;
var_x1 = var(xs1(:));

while abs(1-var_x1)>0.05 && ti<200
    
    x1 = vl_nnconv(x, w1u, w2, ...
        'pad', pad, ...
        'stride', stride, ...
        'dilate', dilate, ...
        opts{:}, ...
        cudnn{:}) ;
    xs1 = vl_nnsumtanh_in(x1) ;
    var_x1 = var(xs1(:));
    w1u = w1u./sqrt(var_x1);
    
    ti = ti+1;
    
end

out = x1;

