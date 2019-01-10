clc
clear all
close all
figure 
hold on


%% Variance and MNOB
% 
% close all
t=1:500; % 500 ms is usually the ASNN simulation time;
         % the average has to be computed within this interval

thMag1 = 0.1;
thTau1 = 15;
etaTau = 50;

theta0 = .5;
theta0r = theta0;

% mnob = [0.1*theta0 0.1*theta0];
mnob = theta0^2; % the asn transfer function considers a scaled version of mf

inputC = .01:.01:1; % input current: signal S to encode

firingratesnn = zeros(length(mnob),length(inputC));
variancesnn = zeros(length(mnob),length(inputC));
meansnn = zeros(length(mnob),length(inputC));
spikesize = 0.5459;
thfactor = 1;

for imnob = 1:length(mnob)
    
    spiketime = [];
    frate = zeros(2,length(inputC));
    tfun = zeros(size(inputC));
    tfunmf = zeros(size(inputC,1),2);
    
    tauSig2 = 2.5;
    a1 = exp (-1/tauSig2);
    theta0 = theta0r/thfactor(imnob);
    for imn = 1:length(inputC)
        
       
        % keep track of assigned theta:
        thetaTrack = ones(size(t))*theta0;
        % define eta kernel
        etaTau = 50;

        % required for exp:
        theta = theta0;
        th0 = theta0;
        th1 = 0;
     
        
        thCorr = 1/(thMag1);
        thMag1c = thMag1*thCorr;
       
        dth1 = exp(-1/(thTau1));
        
        dEta = exp(-1/(etaTau));
        eta = 1;
        % signal:
        tstep = 1;
        input = zeros(1,length(t));
        input(tstep:end) = inputC(imn);
        
        yhat = zeros(size(input));
        spike = zeros(size(input));
        
        
        stimes1a = zeros(size(input));
        y = input;
        
        tauSig2 = .005;
        a2 = exp (-0.001/tauSig2);
        
        epsif = 0;
        
        epsifTrack = ones(size(t));
        
        mISI = 1;
        ISI = 55;
        prevspike = 0;
        
        % figure
        thMag = mnob(imnob);
        
        
        for time=1:length(t)
            
            if time > 300
                theta0 = 0.1;
                thMag = 0.1*0.1;
                spikesize = 0.1244;
            else
                theta0 = theta0r;
                thMag = theta0r^2;
                spikesize = 0.5459;
                
            end
            
            % compute potential as difference between y and yhat
            
            v = y(time) - yhat(time);
            if v>(1*theta/2)
                stimes1a(time) = 1;
                % if a spike: add to approximation
                deltaSpike = time - prevspike;
                
                lenK = length(yhat(time:end));
                % make update to yhat dependent on isi: larger isi, bigger update
%                 if deltaSpike > ISI
%                     isifact = mISI;
%                 else
%                     isifact = 1 + deltaSpike*(mISI-1)/ISI;
%                 end
                isifact  = mISI;
%                 yhat evolves
                dyhat = theta*isifact;
                yhat(time) = yhat(time) + dyhat;
                % theta(time:end) = theta(time:end) + thMag*thKernel(1:lenK);
%                 fmult = theta/theta0;
                fmult = theta;

                
                spike(time) = spike(time) + spikesize;
                
                spiketime = [spiketime time];
                
                % update theta components:
                th1 = th1 + fmult*theta0;
                
                prevspike = time;
            end
            % evolve yhat
            yhat(time+1) = yhat(time)*dEta;
            spike(time+1) = spike(time)*dEta;
            th1 = th1*dth1;
           
            theta = th1 + theta0;
            thetaTrack(time) = theta;
            
            epsif = epsif+(1-a1)*(spike(time)-epsif);
            epsifTrack(time)=epsif;
            
            
        end
        
        
        noS1 = sum(stimes1a(1:end));
        tfun(imn)=noS1;
        
        tfunmf(imn,1)=mean(epsifTrack(end-250:end));
        tfunmf(imn,2)=std(epsifTrack(end-250:end));
    end
    
    
firingratesnn(imnob,:)=tfun/(0.001*t(end));
variancesnn(imnob,:)=tfunmf(:,1);
meansnn(imnob,:)=tfunmf(:,1);

end

% plot(inputC,meansnn(1,:)/max(meansnn(1,:)))
% hold on
% plot(inputC,meansnn(2,:)/max(meansnn(2,:)))

meansnnp = meansnn/meansnn(end);
variancesnn = tfunmf(:,2)'/meansnn(end); %4.6845%12.3577
% h1b=errorbar(inputC,meansnnp(1,:),variancesnn(1,:));
plot(inputC,meansnnp(1,:))
hold on
%
%VL_NNTANH CNN sigmoid nonlinear unit.


%
% plot(inputC,meansnn)
% hold on
%%
% x = inputC;
% figure
% ReLu
% x = x - 0.05;
% x = max(0,x);
dx = 0.0001;
x = -0.5:dx:1;

% h         = single(0.03);
% h         = single(0.0291);

h         = single(1);
m_f       = single(0.1);
tau_gamma = single(15);
tau_eta   = single(50);
theta0    = single(0.1);


c1 = 2*m_f*tau_gamma*tau_gamma;
c2 = 2*theta0*tau_eta*tau_gamma;
c3 = tau_gamma*(m_f*tau_gamma + 2*(m_f + 1)*tau_eta);
c4 = theta0*tau_gamma*tau_eta + theta0*tau_eta*tau_eta;
% h=0.1244
c0 = h/(exp((c1*0.5*theta0 + c2)/(c3*0.5*theta0 + c4)) - 1)- h/2;

y1 = x;
y = x;
% y1(x<theta0/2)= 0;
y1(x<0.0)= 0;
y(x<0.0)= 0;

y1 = max(0,h./(exp((c1*y1 + c2)./(c3*y1 + c4)) - 1) - c0  );
y = max(0,0.1244./(exp((45*y + 150)./((1.6725e+3) * y + 325)) - 1.) - 0.2133);
% y = max(0,0.1244./(exp((45*y + 150)./(1.6725e+3*y + 325)) - 1.) - 1.7149);
% y1 = h./(exp((c1*y1 + c2)./(c3*y1 + c4)) - 1) - c0;

plot(x,y)
hold on
% plot(x,y1)
% hold on
inputC = x;
% plot(inputC,y1/y1(end),'--r')
% plot(inputC,y1/y1(end))


display(y1(end))
display(1/y1(end))
% reluval = max(0,x);
% tanhrect = (tanh(0.25*reluval)+0.45*tanh(0.03*reluval))/1.45;
% plot(x,tanhrect,'--r');
% plot(inputC(2:end),diff(y1/y1(end))/dx)

%%
% elu
% y3 = x;
y3(x>0.1)=y1(x>0.1)/y1(end);
% y3(x<=0.1) = 0.1*exp(x(x<=0.1));
y3(x<=0.1) = 1*(log(1+.1*exp(x(x<=0.1))));

plot(inputC,y3)
% y3(inputC<0) = 0;
% plot(inputC(2:end),diff(y3)/dx)
%%
x = -2:0.01:2;
a = 1*(log(1+.1*exp(x)));
plot(x,a)
%%

x = -20:0.01:2;
a = 0.25*(exp(3*x-1.5));
plot(x,a)
%%

% a = y1/y1(end);
% b = diff(y1/y1(end));
% %%
% close all
% figure
% hold on
% % [ax1,b1,p1] = plotyy(inputC,meansnn(1,:),inputC,firingratesnn(1,:));
% % [ax2,b2,p2] = plotyy(inputC,meansnn(2,:),inputC,firingratesnn(2,:));
% % %
% % firingratenorm = zeros(size(firingratesnn));
% % firingratenorm(1,:) = firingratesnn(1,:)/max(firingratesnn(1,:));
% % firingratenorm(2,:) = firingratesnn(2,:)/max(firingratesnn(2,:));
% 
% variancesnn(1,:) = variancesnn(1,:)/max(meansnn(1,:)); %4.6845%12.3577
% variancesnn(2,:) = variancesnn(2,:)/max(meansnn(2,:)); %12.3577% 35.7537
% 
% 
% meansnnp = zeros(size(meansnn));
% 
% meansnnp(1,:) = meansnn(1,:)/max(meansnn(1,:));
% meansnnp(2,:) = meansnn(2,:)/max(meansnn(2,:));
% % 
% 
% 
% 
% [ax,b,p] = plotyy(inputC,meansnnp,inputC,firingratesnn);
% 
% %
% % hold on
% h1b=errorbar(inputC,meansnnp(1,:),variancesnn(1,:),'LineWidth',2);
% h2b=errorbar(inputC,meansnnp(2,:),variancesnn(2,:),'LineWidth',2);
% % 
% set(p,...
%     'LineWidth',2,...
%     'Marker'          , 'o'         , ...
%     'MarkerSize'      , 8           , ...    
%     'MarkerEdgeColor' , [0 0.4470 0.7410]   , ...
%     'MarkerFaceColor' , [0 0.4470 0.7410]  );
% % 
% 
% set(b,...
%     'LineWidth',2,...    
%     'Marker'          , 'o'         , ...
%     'MarkerSize'      , 5           , ...    
%     'MarkerEdgeColor' , [.2 .2 .2]  , ...
%     'MarkerFaceColor' , [.7 .7 .7]  );
% 
% 
% set(h1b,...
%     'LineWidth',3,...
%     'Marker'          , 'o');
% 
% set(h2b,...
%     'LineWidth',3,...
%     'Marker'          , 'o');
% 
% % c = plot(inputC,(2*sigmf(inputC,[.15/theta0,0]))-1,'LineWidth',2);
% % c1 = plot(inputC,...
% %     tanh((.05/theta0)*inputC)...
% %     ,'LineWidth',2);
% 
% 
% 
% set(ax(1), ...
%     'XLim'        , [0 inputC(end)],...
%   'YLim'        , [0 max(meansnnp(:))],...
%   'Box'         , 'off'     , ...
%   'TickDir'     , 'out'     , ...
%   'TickLength'  , [.02 .02] , ...
%   'XMinorTick'  , 'on'      , ...
%   'YMinorTick'  , 'on'      , ...
%   'YGrid'       , 'off'      , ...
%   'XGrid'       , 'off'      , ...
%   'XTick'       , 0:inputC(end)/10:inputC(end) , ...
%   'YTick'       , 0:0.1:1, ...
%   'LineWidth'   , 2         );
% 
% 
% set(ax(2), ...
%   'YLim'        , [0 max(firingratesnn(:))],...
%   'Box'         , 'off'     , ...
%   'TickDir'     , 'out'     , ...
%   'TickLength'  , [.02 .02] , ...
%   'XMinorTick'  , 'on'      , ...
%   'YMinorTick'  , 'on'      , ...
%   'YGrid'       , 'on'      , ...
%   'XGrid'       , 'off'      , ...
%   'XTick'       ,0:inputC(end)/10:inputC(end), ...
%   'YTick'       , 0:10:max(firingratesnn(:)), ...
%   'YColor'   ,    [0 0.4470 0.7410]  ,...
%   'LineWidth'   , 2         );
% 
% set(ax,'FontSize',24,'FontWeight','Bold');
% 
% 
% hYLabel = ylabel(ax(1),'Approximated S','FontName','Calibri','FontSize',36,'FontWeight','Bold');
% hYLabel2 = ylabel(ax(2),'Firing Rate','FontName','Calibri','FontSize',36,'FontWeight','Bold');
% hXLabel = xlabel('Input Current','FontName','Calibri','FontSize',36,'FontWeight','Bold');
% 
% 
% %%
% cd('<path-to-PowerLawSumExp>')
% 
% 
% lsqOptions = optimset('Display','final','TolFun',1e-10,'MaxFunEvals',100000,'MaxIter',10000);
% xdata = inputC;
% ydata = meansnn(1,:);
% 
% x = getTanhApprox(xdata,ydata, 1, lsqOptions);
% 
% 
% 
% %%
% 
% close all
% 
% figure
% plot(xdata,ydata)
% hold on
% plot(inputC,meansnn(1,:))
% thscale = 1;
% 
% inputCt = [inputC];
% inputCt=sort(inputCt);
% a = x(1)*tanh((x(2)/thscale)*inputCt)+x(3)*tanh((x(4)/thscale)*inputCt);
% plot(inputCt,a,'--')
% %%
% % 
% hold on
% thscale = 1;
% inputCt = 0:0.01:100;
% x = single([0.3282 0.3560 0.6628 2.2374]);
% a = x(1)*tanh((x(2)/thscale)*inputCt)+x(3)*tanh((x(4)/thscale)*inputCt);
% plot(inputCt,a)

% xdatae = inputC;
% ydatae = meansnn(1,:)/4.093;
% plot(xdatae,ydatae)
% aa = ydatae-a;


%% fake plot

a = [99.39 99.36 99.37];
b = [111 69 55];

plot(b,a)
