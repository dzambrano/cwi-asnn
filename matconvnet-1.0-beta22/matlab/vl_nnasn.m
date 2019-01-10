function out = vl_nnasn(x,dzdy)
%VL_NNTANH CNN sigmoid nonlinear unit.
% x = -0.1:0.01:200;

% % ReLu
% x = x + 0.05;
% h         = single(-0.0131);
% m_f       = single(-0.01);
% tau_gamma = single(15);
% tau_eta   = single(50);
% theta0    = single(-0.01);


% 
% h         = single(0.0129);
% m_f       = single(0.01);
% tau_gamma = single(15);
% tau_eta   = single(50);
% theta0    = single(0.01);


% % % 
% h         = single(0.0258);
% m_f       = single(0.02);
% tau_gamma = single(15);
% tau_eta   = single(50);
% theta0    = single(0.02);

% 
% % % 
% h         = single(0.0636);
% m_f       = single(0.05);
% tau_gamma = single(15);
% tau_eta   = single(50);
% theta0    = single(0.05);


% h         = single(0.1004);
% m_f       = single(0.08);
% tau_gamma = single(15);
% tau_eta   = single(50);
% theta0    = single(0.08);



% % 
% h         = single(0.1051);
% m_f       = single(0.1);
% tau_gamma = single(15);
% tau_eta   = single(150);
% theta0    = single(0.1);

% h         = single(0.1099);
% m_f       = single(0.1);
% tau_gamma = single(15);
% tau_eta   = single(100);
% theta0    = single(0.1);

% 
% h         = single(0.1148);
% m_f       = single(0.1);
% tau_gamma = single(15);
% tau_eta   = single(75);
% theta0    = single(0.1);

% 
% h         = single(0.1534);
% m_f       = single(0.1);
% tau_gamma = single(15);
% tau_eta   = single(25);
% theta0    = single(0.1);


% % % 
% h         = single(0.1481);
% m_f       = single(0.12);
% tau_gamma = single(15);
% tau_eta   = single(50);
% theta0    = single(0.12);
% % 
% h         = single(0.2393);
% m_f       = single(0.2);
% tau_gamma = single(15);
% tau_eta   = single(50);
% theta0    = single(0.2);


% % % 
% h         = single(0.3467);
% m_f       = single(0.3);
% tau_gamma = single(15);
% tau_eta   = single(50);
% theta0    = single(0.3);






h         = single(0.1244);
m_f       = single(0.1);
tau_gamma = single(15);
tau_eta   = single(50);
theta0    = single(0.1);


% % 
% h         = single(0.0764);
% m_f       = single(0.1);
% tau_gamma = single(15);
% tau_eta   = single(50);
% theta0    = single(0.1);

% % % 
% h         = single(0.0391);
% m_f       = single(0.05);
% tau_gamma = single(15);
% tau_eta   = single(50);
% theta0    = single(0.05);

% % 
% h         = single(0.2117);
% m_f       = single(0.3);
% tau_gamma = single(15);
% tau_eta   = single(50);
% theta0    = single(0.3);


% h         = single(0.0730);
% m_f       = single(0.2);
% tau_gamma = single(15);
% tau_eta   = single(50);
% theta0    = single(0.02);




% x(x<theta0/2) = 0;
% fixed "training" threshold
x(x<0.0) = 0;


c1 = 2*m_f*tau_gamma*tau_gamma;
c2 = 2*theta0*tau_eta*tau_gamma;
c3 = tau_gamma*(m_f*tau_gamma + 2*(m_f + 1)*tau_eta);
c4 = theta0*tau_gamma*tau_eta + theta0*tau_eta*tau_eta;

c0 = h./(exp((c1*0.5*theta0 + c2)./(c3*0.5*theta0 + c4)) - 1);


gpuMode = isa(x, 'gpuArray') ;

if gpuMode
    
    c1 = gpuArray(c1);
    c2 = gpuArray(c2);
    c3 = gpuArray(c3);
    c4 = gpuArray(c4);
    c0 = gpuArray(c0);
    
end

y = max(0,h./(exp((c1*x + c2)./(c3*x + c4)) - 1) - c0 + h/2);

% trained resnet (theta0=0.1)
% y = max(0,0.1244./(exp((45*x + 150)./(1.6725e+3*x + 325)) - 1.) - 1.7149);
% y = max(0,0.1244./(exp((45*x + 150)./(1.6725e+3*x + 325)) - 1.) - 0.2133); %corrected

if nargin <= 1 || isempty(dzdy)
    out = y ;
else
    
    out = dzdy .* max(0,x);
    
end
