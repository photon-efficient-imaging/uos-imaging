%% Photon-Efficient Depth Imaging Using a Union-of-Subspaces Model
%% D. Shin, J. H. Shapiro, V. K Goyal
%% call experimental data

clc; clear; close all;

bin_start = 2000;
bin_end = 6000;
bin_length = 5;
%
fileName = 'data_mannequin_face';
binning = bin_start:bin_length:bin_end;
m = length(binning);

load(fileName);
[nr,nc] = size(arrivalTimes);

fprintf(['# define parameters \n'])
fprintf(['  face dim  = ' mat2str(size(arrivalTimes)) '\n'])
fprintf(['  bin siz   = ' num2str(bin_length) '\n'])
fprintf(['  hist siz  = ' num2str(m) '\n'])

% calibration of SBR
dats_hist = [];
fprintf('\n')
fprintf('# calibrate (sbr only) \n')
for i=1:nr
    for j=1:nc
        dats = arrivalTimes{i,j};
        dats(dats<bin_start) = [];
        dats(dats>bin_end) = [];
        dats_hist = [dats_hist; dats(1)];
    end
end
fprintf('done.. \n')

b_calibrate = 50;

m_hist = 200;
[y,x] = hist(dats_hist,m_hist);
strength_total = sum(y);
strength_background = m_hist*b_calibrate;
strength_signal = strength_total-strength_background;
sbr = strength_signal/strength_background;
sbr_peak_to_peak = (max(y)-b_calibrate)/b_calibrate;
fprintf(['S         = ' num2str(strength_signal) '\n'])
fprintf(['B         = ' num2str(strength_background) '\n'])
fprintf(['SBR       = ' num2str(sbr) '\n'])

% compute kernel matrix

fprintf('\n')
fprintf('# compute kernel \n')
t = 1:1:m;
rms_pulsewidth = 45;
rms_pulselength_cm = rms_pulsewidth*(8e-12)*(3e8)*100;
sigs = rms_pulsewidth/bin_length;
f = @(x) exp(-abs(x).^2/(2*sigs^2)); % peak is 1
% Generate dictionary
S = zeros(m,m);
for i=1:m        
    s = f(t-t(i));
    s = s/max(s);
    S(:,i) = s';
end
logS = log(S);
logS = logS-max(logS(:));
min_logS = min(logS(~isinf(-logS)));
logS(isinf(-logS)) = min_logS;

fprintf(['RMS PW     = ' num2str(sigs) '\n']);
fprintf(['RMS length = ' num2str(rms_pulselength_cm) ' [cm]\n']);

% run estimators

fprintf('\n')
fprintf('# run experiments \n')

num_photons_per_pixel = 15; % num photons per pixel

min_time = 3550;
max_time = 3700;
dels = 1e-4; % number of 
ite_max = 10; % max iteration num
dwnsmpl = 1; % downsampling degree of image

A = [S,ones(m,1)];

nr_fin = length(1:dwnsmpl:nr);
nc_fin = length(1:dwnsmpl:nr);
%
load([fileName '_truth']);
D_truth = cell2mat(D_true);
M = (D_truth>min_time)&(D_truth<max_time);
M = M(1:dwnsmpl:end,1:dwnsmpl:end);
%
D_ML_B0 = zeros(nr_fin,nc_fin);
D_ML_B_opt = zeros(nr_fin,nc_fin);
D_UOS = zeros(nr_fin,nc_fin);
% store some metadata for UOS
B_UOS = zeros(nr_fin,nc_fin);
ite_UOS = zeros(nr_fin,nc_fin);
y_all = zeros(1,length(binning));
fprintf(['photons-per-pixel = ' num2str(num_photons_per_pixel) '\n']);
for i=1:1:nr_fin
    if(mod(i,10)==0)
        fprintf([' row ' num2str(i) ' out of ' num2str(nr_fin) '\n'])
    end
    for j=1:1:nc_fin       
        dats = arrivalTimes{1+((i-1)*dwnsmpl),1+((j-1)*dwnsmpl)};
        dats(dats<bin_start) = [];
        dats(dats>bin_end) = [];
        % get a photon count signal
        [y,inds] = hist(dats(1:min(num_photons_per_pixel,length(dats))),binning);
        % get an accumulated histogram
        if(M(i,j)==1)
            y_all = y_all + y;
        end
        % estimator #1: log-matcher
        [vals,inds_ml_B0] = max(logS*y');
        D_ML_B0(i,j) = binning(inds_ml_B0(1));
        % estimator #2: UOS
        [sol_UOS,ite_conv] = COSAMP_UOS(y',A,dels,ite_max);
        D_UOS(i,j) = binning(find(sol_UOS(1:m)));
        B_UOS(i,j) = sol_UOS(end);
        ite_UOS(i,j) = ite_conv;
    end
end
fprintf('Finished processing ... \n');

num_ite_average = mean(ite_UOS(:));
fprintf(['ite average = ' num2str(num_ite_average) '\n'])

figure;
subplot(131); imagesc(D_truth,[min_time,max_time]);
axis image; colorbar; colormap('jet');
subplot(132); imagesc(D_ML_B0,[min_time,max_time]);
axis image; colorbar; colormap('jet');
subplot(133); imagesc(D_UOS,[min_time,max_time]);
axis image; colorbar; colormap('jet');
%
