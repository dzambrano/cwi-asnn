clc
clear all
close all
% figure 
% hold on

cd(asnn_root)

%% Variance and MNOB
% 
% close all
t=1:3000;

thMag1 = .1;
thTau1 = 15;
etaTau = 25;

theta0v = [0.5 0.1];

mnob = [theta0v(1)*theta0v(1) theta0v(2)*theta0v(2)];
% mnob = [1*theta0v(1) 0.5*theta0v(2)];


inputC = .0:.2:5;

firingratesnn = zeros(length(mnob),length(inputC));
variancesnn = zeros(length(mnob),length(inputC));
meansnn = zeros(length(mnob),length(inputC));


for imnob = 1:length(mnob)
    
    
    spiketime = [];
    frate = zeros(2,length(inputC));
    tfun = zeros(size(inputC));
    tfunmf = zeros(size(inputC,1),2);
    
    tauSig2 = 2.5;
    a1 = exp (-1/tauSig2);
    
    theta0 = theta0v(imnob);
    
    for imn = 1:length(inputC)
        
        % keep track of assigned theta:
        thetaTrack = ones(size(t))*theta0;
        
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
            
            
            v = y(time) - yhat(time);
            if v>(1*theta/2)
                stimes1a(time) = 1;
                % if a spike: add to approximation
                deltaSpike = time - prevspike;
                
                lenK = length(yhat(time:end));
                
                isifact  = mISI;

                dyhat = theta*isifact;
                yhat(time) = yhat(time) + dyhat;

                
                fmult = theta/theta0;

                
                spike(time) = spike(time) + 1;
                spiketime = [spiketime time];
                
                % update theta components:
                th1 = th1 + fmult*thMag;
                
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
        
        
        noS1 = sum(stimes1a(1500:2500));
        tfun(imn)=noS1;
        
        tfunmf(imn,1)=mean(epsifTrack(1500:2500));
        tfunmf(imn,2)=std(epsifTrack(1500:2500));
    end
    
    
firingratesnn(imnob,:)=tfun;
variancesnn(imnob,:)=tfunmf(:,2);
meansnn(imnob,:)=tfunmf(:,1);

end

% h1b=errorbar(inputC,meansnn(1,:),variancesnn(1,:),'LineWidth',2);
% h1b=errorbar(inputC,meansnn(2,:),variancesnn(2,:),'LineWidth',2);
% 
% plot(inputC,meansnn(1,:)/max(meansnn(1,:)))
% hold on
% plot(inputC,meansnn(2,:)/max(meansnn(2,:)))

% 
% plot(inputC,meansnn(1,:)/meansnn(1,6))
% hold on
% plot(inputC,meansnn(2,:))


% hold on
%
%VL_NNTANH CNN sigmoid nonlinear unit.


%% split figure

close all
figure('units','normalized','position',[.1 .1 .8 .8])
hold on

% variancesnnp(1,:) = variancesnn(1,:)/max(meansnn(1,:)); %4.6845%12.3577
% variancesnnp(2,:) = variancesnn(2,:)/max(meansnn(2,:)); %12.3577% 35.7537

variancesnnp(1,:) = variancesnn(1,:)/1.8318; %4.6845%12.3577
variancesnnp(2,:) = variancesnn(2,:)/8.0368; %12.3577% 35.7537


meansnnp = zeros(size(meansnn));

% meansnnp(1,:) = meansnn(1,:)/max(meansnn(1,:));
% meansnnp(2,:) = meansnn(2,:)/max(meansnn(2,:));

meansnnp(1,:) = meansnn(1,:)/1.8318;
meansnnp(2,:) = meansnn(2,:)/8.0368;

% 

% [ax,b,p] = plotyy(inputC,meansnnp,inputC,firingratesnn);

%

subplot(2,1,1), hold on

ax = gca;
ax.ColorOrderIndex = 3;


h1b=errorbar(inputC,meansnnp(1,:),variancesnnp(1,:),'LineWidth',2);
h2b=errorbar(inputC,meansnnp(2,:),variancesnnp(2,:),'LineWidth',2);


set(ax, ...
    'XLim'        , [0 5],...
  'YLim'        , [0 3],...
  'TickDir'     , 'in'     , ...
  'TickLength'  , [.02 .02] , ...
  'XMinorTick'  , 'on'      , ...
  'YMinorTick'  , 'on'      , ...
  'Box'         , 'off'     , ... 
  'YGrid'       , 'off'      , ...
  'XGrid'       , 'off'      , ...
  'YTick'       , 0:1:3 , ...
  'XTick'       , 0:1:5 , ...
  'YMinorTick'  , 'on'      , ...
  'LineWidth'   , 2         );


% axis(ax,'square')

% Defaults for this blog post
width = .4;     % Width in inches
height = .3;    % Height in inches
alw = 0.75;    % AxesLineWidth
fsz = 11;      % Fontsize 
lw = 1.5;      % LineWidth
msz = 8;       % MarkerSize

pos = get(ax, 'Position');
set(ax, 'Position', [pos(1) pos(2) width, height]);


set(ax,'FontSize',28,'FontWeight','Normal');
hYLabel = ylabel(ax(1),'\DeltaS_{post}','FontName','Helvetica','FontSize',24,'FontWeight','Normal');

hle = legend('Low FR','High FR','Location','NorthWest');
legend boxoff
subplot(2,1,2), hold on

ax = gca;
ax.ColorOrderIndex = 3;


hs2 = plot(inputC,firingratesnn,'LineWidth',2);


set(ax, ...
     'XLim'        , [0 5],...
  'YLim'        , [0 max(firingratesnn(:))],...  
  'TickDir'     , 'in'     , ...
  'TickLength'  , [.02 .02] , ...
  'XMinorTick'  , 'on'      , ...
  'YMinorTick'  , 'on'      , ...
  'Box'         , 'off'     , ... 
  'YGrid'       , 'off'      , ...
  'XGrid'       , 'off'      , ...
  'XTick'       , 0:1:5 , ...
  'YTick'       , 0:100:max(firingratesnn(:)), ...    
  'YMinorTick'  , 'on'      , ...
  'LineWidth'   , 2         );


ax = gca;
pos = get(ax, 'Position');
set(ax, 'Position', [pos(1) pos(2) width, height]);

hXLabel = xlabel(ax,'S_{pre}','FontName','Helvetica','FontSize',24,'FontWeight','Normal');
hYLabel2 = ylabel(ax,'Firing Rate','FontName','Helvetica','FontSize',24,'FontWeight','Normal');

set(ax,'FontSize',28,'FontWeight','Normal');

% 
%%
% set(b,...
%     'LineWidth',2,...    
%     'Marker'          , 'o'         , ...
%     'MarkerSize'      , 5           , ...    
%     'MarkerEdgeColor' , [.2 .2 .2]  , ...
%     'MarkerFaceColor' , [.7 .7 .7]  );


% set(h1b,...
%     'LineWidth',3,...
%     'Marker'          , 'o');
% 
% set(h2b,...
%     'LineWidth',3,...
%     'Marker'          , 'o');

% c = plot(inputC,(2*sigmf(inputC,[.15/theta0,0]))-1,'LineWidth',2);
% c1 = plot(inputC,...
%     tanh((.05/theta0)*inputC)...
%     ,'LineWidth',2);



set(ax(1), ...
    'XLim'        , [0 inputC(end)],...
  'YLim'        , [0 4],...
  'Box'         , 'off'     , ...
  'TickDir'     , 'out'     , ...
  'TickLength'  , [.02 .02] , ...
  'XMinorTick'  , 'on'      , ...
  'YMinorTick'  , 'on'      , ...
  'YGrid'       , 'off'      , ...
  'XGrid'       , 'off'      , ...
  'XTick'       , 0:inputC(end)/10:inputC(end) , ...
  'YTick'       , 0:0.5:4, ...
  'LineWidth'   , 2         );


set(ax(2), ...
  'YLim'        , [0 max(firingratesnn(:))],...
  'Box'         , 'off'     , ...
  'TickDir'     , 'out'     , ...
  'TickLength'  , [.02 .02] , ...
  'XMinorTick'  , 'on'      , ...
  'YMinorTick'  , 'on'      , ...
  'YGrid'       , 'off'      , ...
  'XGrid'       , 'off'      , ...
  'XTick'       ,0:inputC(end)/10:inputC(end), ...
  'YTick'       , 0:50:max(firingratesnn(:)), ...
  'YColor'   ,    [0 0.4470 0.7410]  ,...
  'LineWidth'   , 2         );

set(ax,'FontSize',24,'FontWeight','Normal');
axis(ax,'square')


hXLabel = xlabel(ax(1),'S_{pre}','FontName','Helvetica','FontSize',24,'FontWeight','Normal');
hYLabel = ylabel(ax(1),'\DeltaS_{post}','FontName','Helvetica','FontSize',24,'FontWeight','Normal');
hYLabel2 = ylabel(ax(2),'Firing Rate','FontName','Helvetica','FontSize',24,'FontWeight','Normal');

aop=get(gca,'OuterPosition');
set(gca,'OuterPosition',[aop(1) aop(2)+0.1 aop(3)-0.05 aop(4)-0.05])








%%
% 
% close all
x = 0.01:0.01:5;

% h         = single(0.03);
% h         = single(0.0291);
h         = single(1);
m_f       = single(0.1);
tau_gamma = single(15);
tau_eta   = single(25);
theta0    = single(0.1);

c1 = 2*m_f*tau_gamma*tau_gamma;
c2 = 2*theta0*tau_eta*tau_gamma;
c3 = tau_gamma*(m_f*tau_gamma + 2*(m_f + 1)*tau_eta);
c4 = theta0*tau_gamma*tau_eta + theta0*tau_eta*tau_eta;

c0 = h/(exp((c1*0.5*theta0 + c2)/(c3*0.5*theta0 + c4)) - 1);

y1 = x;
y1(x<theta0/2)= 0;
y1 = max(0,h./(exp((c1*y1 + c2)./(c3*y1 + c4)) - 1) - c0 + h/2);
% 
% y = x;
% y(x<0.05)= 0;
% y(x>=0.05) = h./(exp((c1*y(x>=0.05) + c2)./(c3*y(x>=0.05) + c4)) - 1) - c0;

% hold on
% plot(inputC,y,'--g')
% figure
hold on
% inputC = x;
% plot(inputC,y1/y1(end),'--r')

% plot(x,y1/y1(100),'r','LineWidth',2)
plot(x,y1/y1(100),'LineWidth',2)
ax = gca;
set(ax, ...
    'XLim'        , [0 5],...
  'YLim'        , [0 3],...
  'Box'         , 'off'     , ...
  'TickDir'     , 'in'     , ...
  'TickLength'  , [.02 .02] , ...
  'XMinorTick'  , 'on'      , ...
  'YMinorTick'  , 'on'      , ...
  'YGrid'       , 'off'      , ...
  'XGrid'       , 'off'      , ...
  'XTick'       , 0:1:5 , ...
  'YTick'       , 0:1:4, ...
  'LineWidth'   , 2         );

set(ax,'FontName','Helvetica','FontSize',28,'FontWeight','Normal');
hXLabel = xlabel(ax,'S_{pre}','FontName','Helvetica','FontSize',28,'FontWeight','Normal');
hYLabel = ylabel(ax,'\DeltaS_{post}','FontName','Helvetica','FontSize',28,'FontWeight','Normal');

axis(ax,'square')
% 
% reluval = max(0,x);
% tanhrect = (tanh(0.25*reluval)+0.45*tanh(0.03*reluval))/1.45;
% % plot(x,tanhrect,'--r');


display(y1(100))
display(1/y1(100))


%%
close all
figure('units','normalized','position',[.1 .1 .8 .8])
hold on

% variancesnnp(1,:) = variancesnn(1,:)/max(meansnn(1,:)); %4.6845%12.3577
% variancesnnp(2,:) = variancesnn(2,:)/max(meansnn(2,:)); %12.3577% 35.7537

variancesnnp(1,:) = variancesnn(1,:)/1.8318; %4.6845%12.3577
variancesnnp(2,:) = variancesnn(2,:)/8.0368; %12.3577% 35.7537


meansnnp = zeros(size(meansnn));

% meansnnp(1,:) = meansnn(1,:)/max(meansnn(1,:));
% meansnnp(2,:) = meansnn(2,:)/max(meansnn(2,:));

meansnnp(1,:) = meansnn(1,:)/1.8318;
meansnnp(2,:) = meansnn(2,:)/8.0368;

% 


[ax,b,p] = plotyy(inputC,meansnnp,inputC,firingratesnn);

%
% hold on
h1b=errorbar(inputC,meansnnp(1,:),variancesnnp(1,:),'LineWidth',2);
h2b=errorbar(inputC,meansnnp(2,:),variancesnnp(2,:),'LineWidth',2);
% 
set(p,...
    'LineWidth',2,...
    'Marker'          , 'o'         , ...
    'MarkerSize'      , 4           , ...    
    'MarkerEdgeColor' , [0 0.4470 0.7410]   , ...
    'MarkerFaceColor' , [0 0.4470 0.7410]  );


% set(b,...
%     'LineWidth',2,...    
%     'Marker'          , 'o'         , ...
%     'MarkerSize'      , 5           , ...    
%     'MarkerEdgeColor' , [.2 .2 .2]  , ...
%     'MarkerFaceColor' , [.7 .7 .7]  );


% set(h1b,...
%     'LineWidth',3,...
%     'Marker'          , 'o');
% 
% set(h2b,...
%     'LineWidth',3,...
%     'Marker'          , 'o');

% c = plot(inputC,(2*sigmf(inputC,[.15/theta0,0]))-1,'LineWidth',2);
% c1 = plot(inputC,...
%     tanh((.05/theta0)*inputC)...
%     ,'LineWidth',2);



set(ax(1), ...
    'XLim'        , [0 inputC(end)],...
  'YLim'        , [0 4],...
  'Box'         , 'off'     , ...
  'TickDir'     , 'out'     , ...
  'TickLength'  , [.02 .02] , ...
  'XMinorTick'  , 'on'      , ...
  'YMinorTick'  , 'on'      , ...
  'YGrid'       , 'off'      , ...
  'XGrid'       , 'off'      , ...
  'XTick'       , 0:inputC(end)/10:inputC(end) , ...
  'YTick'       , 0:0.5:4, ...
  'LineWidth'   , 2         );


set(ax(2), ...
  'YLim'        , [0 max(firingratesnn(:))],...
  'Box'         , 'off'     , ...
  'TickDir'     , 'out'     , ...
  'TickLength'  , [.02 .02] , ...
  'XMinorTick'  , 'on'      , ...
  'YMinorTick'  , 'on'      , ...
  'YGrid'       , 'off'      , ...
  'XGrid'       , 'off'      , ...
  'XTick'       ,0:inputC(end)/10:inputC(end), ...
  'YTick'       , 0:50:max(firingratesnn(:)), ...
  'YColor'   ,    [0 0.4470 0.7410]  ,...
  'LineWidth'   , 2         );

set(ax,'FontSize',24,'FontWeight','Normal');
axis(ax,'square')


hXLabel = xlabel(ax(1),'S_{pre}','FontName','Helvetica','FontSize',24,'FontWeight','Normal');
hYLabel = ylabel(ax(1),'\DeltaS_{post}','FontName','Helvetica','FontSize',24,'FontWeight','Normal');
hYLabel2 = ylabel(ax(2),'Firing Rate','FontName','Helvetica','FontSize',24,'FontWeight','Normal');

aop=get(gca,'OuterPosition');
set(gca,'OuterPosition',[aop(1) aop(2)+0.1 aop(3)-0.05 aop(4)-0.05])


% print -depsc2 TR1.eps %
% print('FillPageFigure','-dpdf','-fillpage')
%%
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