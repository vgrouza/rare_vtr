function t1map_vector = rarevtr_t1fit(nii_dir, mask_dir)
% computes the masked T1 map for RARE_VTR in vivo scans.
% 
% as per documentation, we use load_untouch/save_untouch because we want
% only the raw img data without header trasnforms applied.

addpath(genpath('/data/rudko/vgrouza/invivomouse/niftitools'));

rare_vtr = load_untouch_nii(nii_dir);
rare_vtr_img = double(rare_vtr.img);
mask = load_untouch_nii(mask_dir);
mask_img = double(mask.img);

NX = size(rare_vtr_img, 1);
NY = size(rare_vtr_img, 2);
NZ = size(rare_vtr_img, 3);

% preset the repetition times (TRs) in msec
if size(rare_vtr_img, 4) == 6
    rep_times = [8000, 2400, 1480, 940, 650, 418]; 

elseif size(rare_vtr_img, 4) == 7
    rep_times = [8000, 3600, 2400, 1480, 940, 650, 501.1];
else 
    disp('Incorrect VTR vector');
    return;
end   

% if segmentation and VTR setting are good, proceed and apply binary mask.
rare_vtr_img = rare_vtr_img.*mask_img;
mask.img = squeeze(rare_vtr_img(:,:,:,1));
save_untouch_nii(mask, fullfile(fileparts(nii_dir), 'masked_first_vtr.nii.gz'));

rare_vtr_img = reshape(rare_vtr_img, [NX*NY*NZ, length(rep_times)]);
mask_img = reshape(mask_img, [NX*NY*NZ, 1]);

% Set the options parameter which controls (among other things) the number
% of iterations for lsqcurvefit
OPTIONS = optimoptions('lsqcurvefit', ...
                       'Display','off',...
                       'TolFun',1e-3, ...
                       'MaxFunEvals',100,'MaxIter',100);
tic;               
t1map_img = zeros(size(mask_img));
fitted_params_vec = zeros(length(mask_img),2);
parfor i = 1:length(mask_img)
    if mask_img(i)~=0
        sig_intensity = rare_vtr_img(i,:);        
        % simple mono-exponential signal model
        signal_model = @(x,xdata) x(1)*(1-exp(-rep_times/x(2)));
        starting_guess = [max(sig_intensity) 1800];
        % carry out the fit, but for the time being we may ignore the
        % parameters computed in the error structure (SSE, GOF, R2, 0.95CI)
        [fitted_params,error] = lsqcurvefit(signal_model,...
                                            starting_guess, ...
                                            rep_times,...
                                            sig_intensity,...
                                            [0, 0],...
                                            [Inf, 10000], ...
                                            OPTIONS);  
        exponential_fit = fitted_params(2);                                                                       
        t1map_img(i,1) = exponential_fit;
        fitted_params_vec(i,:) = fitted_params;        
    end    
end
toc;

% % Visualize fit for a random pixel (for documentation)
% times = linspace(rep_times(1), rep_times(end));
% sig_mod = @(x,xdata)x(1)*(1-exp(-times/x(2)));
% figure(1); hold on; 
%     pixel_num = 148627;
%     scatter(rep_times, rare_vtr_img(pixel_num, :), 'k'); 
%     plot(times, sig_mod(fitted_params_vec(pixel_num,:), times), 'r')
%     legend('Measured Data','Fitted Signal Model','Location','best')
%     xlabel('Repetition Time (msec)'); ylabel('Signal Intensity');
%     box on; grid on; title('Voxel-wise Fitting of Signal Model')

t1map_vector = t1map_img(logical(mask_img));
t1map_img = reshape(t1map_img, [NX, NY, NZ]);
mask.img = t1map_img;

save_untouch_nii(mask, fullfile(fileparts(nii_dir), 't1_map_corrected.nii.gz'));


end
