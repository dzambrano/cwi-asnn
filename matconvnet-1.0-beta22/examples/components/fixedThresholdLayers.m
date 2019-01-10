
function res_t = fixedThresholdLayers(res_,net_rl)


res_t = struct;

datapos = res_(1).x>0;
% nntrainmin = min(res_(1).x(datapos));
nntrainmin = 0.1;


for irt = 1:numel(res_)-1
   
   if strcmp(net_rl.layers{irt}.type,'sumtanh')
        for in_m = 1:size(res_(irt).x,3)
            for in_l1 = 1:size(res_(irt).x,1)
%                 in_l1
                for in_l2 = 1:size(res_(irt).x,2)

                    res_t(irt).fitWDistA(in_l1,in_l2,in_m) = nntrainmin;
                    res_t(irt).fitWDistB(in_l1,in_l2,in_m) = 1;
                    
                end
            end
        end
        
    
    
    elseif strcmp(net_rl.layers{irt}.type,'pool')
        if strcmp(net_rl.layers{irt}.method,'max')
        for in_m = 1:size(res_(irt+1).x,3)
            
            fprintf('Processing layer %2.0f, map %3.0f.\n ',[irt, in_m] );
            
            for in_l1 = 1:size(res_(irt+1).x,1)
                for in_l2 = 1:size(res_(irt+1).x,2)
                    
                    res_t(irt).fitWDistA(in_l1,in_l2,in_m) = nntrainmin;
                    res_t(irt).fitWDistB(in_l1,in_l2,in_m) = 1;
                    
                end
            end
        end
        end
        
   
    end
end



end