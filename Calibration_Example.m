
%Here we propose the calibration method for any desired coverage level. We
%first take dataset (e.g., White Wine) and partition into 3 parts: Train
%dataset (70%), calibration dataset(15%) and test dataset(15%). After
%finishing training on Z-GT2-FLS model for coverage level 99 percent. We implement derivate
%free search algorithm for desired coverage level.


%%
clc;clear;
close all;
seed = 0;
rng(seed)


%% White Wine

dataset_name = 'wine';

load("/home/yusuf/Desktop/fuzzy/Fuzzy-Code-Base-master/dataset/Datasets/whitewine.mat");

data = [x y];

training_num = 3429;
calibrated_num = 734;

mbs = 64; %mini batch size
learnRate = 0.001;
lr = learnRate;
number_of_epoch = 100;
%%
current_path = pwd;

number_mf = 10; % number of rules == number of membership functions

number_inputs = min(size(x));
number_outputs = min(size(y));



input_membership_type = "gaussmf";

input_type ="H";
output_membership_type = "linear";
type_reduction_method = "KM";


delta = struct;
delta.delta1 = dlarray(zeros(1, 1));
delta.delta4 = dlarray(ones(1, 1));
alpha = [0.01 0.5 1];

alpha = permute(alpha, [4, 3, 1, 2]);
alpha_rev = (permute(alpha, [1, 4, 2, 3])) / sum(alpha); %to be used in LA1 and LA2

type_reduction_method_list = ["KM"];

gradDecay = 0.9;
sqGradDecay = 0.999;

plotFrequency = 50;
learnRate = lr;

averageGrad = [];
averageSqGrad = [];

%%

if type_reduction_method == "KM"
    u = int2bit(0:(2^number_mf)-1,number_mf);
else
    u = 0;
end

how_to_save = append("GT-2-",input_type,"-",output_membership_type,"-",type_reduction_method,"_gauss_LA1","-", dataset_name,"-rng",string(seed))
%% Normalization upfront ------------------------------

[xn,input_mean,input_std] = zscore_norm(x);
[yn,output_mean,output_std] = zscore_norm(y);

data = [xn yn];
%% split by number ------------------------------

data_size = max(size(data));
test_num = data_size-training_num - calibrated_num;

idx = randperm(data_size);

Training_temp = data(idx(1:training_num),:);
Calibrated_temp = data(idx(training_num + 1 : training_num + calibrated_num), :);
Testing_temp = data(idx(training_num+calibrated_num+1:end),:);

%% ------------------------------

%training data
Train.inputs = reshape(Training_temp(:,1:number_inputs)', [1, number_inputs, training_num]); % traspose come from the working mechanism of the reshape, so it is a must
Train.outputs = reshape(Training_temp(:,(number_inputs+1:end))', [1, number_outputs, training_num]);

Train.inputs = dlarray(Train.inputs);
Train.outputs = dlarray(Train.outputs);

%testing data
Test.inputs = reshape(Testing_temp(:,1:number_inputs)', [1, number_inputs, test_num]);
Test.outputs = reshape(Testing_temp(:,(number_inputs+1:end))', [1, number_outputs, test_num]);

%calibrated data

Calibrated.inputs = reshape(Calibrated_temp(:,1:number_inputs)', [1, number_inputs, calibrated_num]);
Calibrated.outputs = reshape(Calibrated_temp(:,(number_inputs+1:end))', [1, number_outputs, calibrated_num]);


%% init

Learnable_parameters = initialize_Glorot_GT2(Train.inputs, input_type, Train.outputs, output_membership_type, number_mf);
prev_learnable_parameters = Learnable_parameters;
%% rng reset
rng(seed)

%% denormalizing for plotting

yTrue_train = reshape(Train.outputs, [1, max(size(Train.inputs))]);
yTrue_test = reshape(Test.outputs, [1, max(size(Test.inputs))]);

%%

number_of_iter_per_epoch = floorDiv(training_num, mbs);

number_of_iter = number_of_epoch * number_of_iter_per_epoch;
global_iteration = 1;

for epoch = 1: number_of_epoch
    

    [batch_inputs, batch_targets] = create_mini_batch(Train.inputs, Train.outputs, training_num);


    for iter = 1:number_of_iter_per_epoch

        [mini_batch_inputs, targets] = call_batch(batch_inputs, batch_targets,iter,mbs);


        [loss, gradients, ~, ~, ~] = dlfeval(@GT2_fismodelLoss, mini_batch_inputs ,...
            number_inputs, targets,number_outputs, number_mf, mbs, Learnable_parameters, output_membership_type,...
            input_membership_type,input_type,type_reduction_method,u, alpha, delta, alpha_rev);
        
        [Learnable_parameters, averageGrad, averageSqGrad] = adamupdate(Learnable_parameters, gradients, averageGrad, averageSqGrad,...
            epoch, learnRate, gradDecay, sqGradDecay);

    end


    %testing in each epoch
    [yPred_test_lower, yPred_test_upper, yPred_test] = GT2_fismodel_LA1_new(Test.inputs, number_mf, number_inputs,number_outputs,length(Test.inputs), Learnable_parameters, output_membership_type,input_membership_type,input_type,type_reduction_method,u, alpha, delta, alpha_rev);

    yPred_test = reshape(yPred_test, [1, max(size(Test.inputs))]);
    yPred_test_upper = reshape(yPred_test_upper, [1, max(size(Test.inputs))]);
    yPred_test_lower = reshape(yPred_test_lower, [1, max(size(Test.inputs))]);
        
    iter_plot_T2(epoch,plotFrequency,loss,yTrue_test, yPred_test, yPred_test_upper, yPred_test_lower);


end

%% Inference
[yPred_train_lower, yPred_train_upper, yPred_train] = GT2_fismodel_LA1_new(Train.inputs, number_mf, number_inputs,number_outputs,length(Train.inputs), Learnable_parameters, output_membership_type,input_membership_type,input_type,type_reduction_method,u,alpha, delta, alpha_rev);
[yPred_test_lower, yPred_test_upper, yPred_test] = GT2_fismodel_LA1_new(Test.inputs, number_mf, number_inputs,number_outputs,length(Test.inputs), Learnable_parameters, output_membership_type,input_membership_type,input_type,type_reduction_method,u, alpha, delta, alpha_rev);


yPred_train = reshape(yPred_train, [1, max(size(Train.inputs))]);
yPred_train_upper = reshape(yPred_train_upper, [1, max(size(Train.inputs))]);
yPred_train_lower = reshape(yPred_train_lower, [1, max(size(Train.inputs))]);

yPred_test = reshape(yPred_test, [1, max(size(Test.inputs))]);
yPred_test_upper = reshape(yPred_test_upper, [1, max(size(Test.inputs))]);
yPred_test_lower = reshape(yPred_test_lower, [1, max(size(Test.inputs))]);




train_RMSE = rmse(yPred_train, yTrue_train);
test_RMSE = rmse(yPred_test, yTrue_test);


PI_train = PICP(yTrue_train, yPred_train_lower, yPred_train_upper);
PI_test = PICP(yTrue_test, yPred_test_lower, yPred_test_upper);
PI_NAW_train = PINAW(yTrue_train, yPred_train_lower, yPred_train_upper);
PI_NAW_test = PINAW(yTrue_test, yPred_test_lower, yPred_test_upper);

%% Inference on Calibrated
[yPred_calib_lower, yPred_calibrated_upper, yPred_calibrated] = GT2_fismodel_LA1_new(Calibrated.inputs, number_mf, number_inputs,number_outputs,length(Calibrated.inputs), Learnable_parameters, output_membership_type,input_membership_type,input_type,type_reduction_method,u,alpha, delta, alpha_rev);

yPred_calibrated = reshape(yPred_calibrated, [1, max(size(Calibrated.inputs))]);
yPred_calibrated_upper = reshape(yPred_calibrated_upper, [1, max(size(Calibrated.inputs))]);
yPred_calib_lower = reshape(yPred_calib_lower, [1, max(size(Calibrated.inputs))]);


Calib_RMSE = rmse(yPred_calibrated, yTrue_calib);


PI_Calib = PICP(yTrue_calib, yPred_calib_lower, yPred_calibrated_upper);
PI_NAW_Calib = PINAW(yTrue_calib, yPred_calib_lower, yPred_calibrated_upper);

%% Post processing for the model calib data

alpha_to_be_used = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];
alpha_to_be_used = permute(alpha_to_be_used, [4, 3, 1, 2]);

[yPred_calib_lower, yPred_calib_upper] = GT2_fismodel_LA1_per_alpha(Calibrated.inputs, number_mf, number_inputs,number_outputs,length(Calibrated.inputs), Learnable_parameters, output_membership_type,input_membership_type,input_type,type_reduction_method,u, alpha_to_be_used, delta);

yPred_calib_upper = permute(yPred_calib_upper, [1 3 2]);
yPred_calib_lower = permute(yPred_calib_lower, [1 3 2]);

PI_calib_post_values = [];
PINAW_calib_post_values = [];

for i = 1:size(yPred_calib_lower, 1)
    PI_calib_post = PICP(yTrue_calib, yPred_calib_lower(i, :), yPred_calib_upper(i, :));
    PINAW_calib_post = PINAW(yTrue_calib, yPred_calib_lower(i,:), yPred_calib_upper(i, :));
    PI_calib_post_values = [PI_calib_post, PI_calib_post_values];
    PINAW_calib_post_values = [PINAW_calib_post, PINAW_calib_post_values];
end

%% Drawing calib
alpha_to_be_used = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];
alpha_to_be_used = permute(alpha_to_be_used, [4, 3, 1, 2]);

figure
alpha_to_be_used = permute(flip(alpha_to_be_used), [1 4 2 3]);

% % Plot the Calibration Curve
plot(PI_calib_post_values, alpha_to_be_used, 'LineWidth', 2);
xlabel("PI Coverage (\phi)", 'FontWeight', 'bold');  % Bold x-axis label
ylabel("\alpha-planes", 'FontWeight', 'bold');  % Bold y-axis label

%% derivative_free_search_algorithm for optimal Alpha

step_size = 0.1;  % Initial step size
exploration_factor = 0.5;  % Factor by which to reduce the step size
tolerance = 0.05;  % Convergence tolerance
target = 95;  % Target value we want f(alpha) to reach
alpha0 = 0.40;

alpha_opt = derivative_free_search_algorithm(Calibrated.inputs, yTrue_calib, number_mf, number_inputs, number_outputs, length(Calibrated.inputs), Learnable_parameters, output_membership_type,input_membership_type,input_type,type_reduction_method,u, alpha0, delta, 1, step_size, exploration_factor, tolerance, target);

%% Post processing for the model test data with calibrated
% 
alpha_to_be_used = [alpha_opt];

alpha_to_be_used = permute(alpha_to_be_used, [4, 3, 1, 2]);

[yPred_testsp_lower, yPred_testsp_upper] = GT2_fismodel_LA1_per_alpha(Test.inputs, number_mf, number_inputs,number_outputs,length(Test.inputs), Learnable_parameters, output_membership_type,input_membership_type,input_type,type_reduction_method,u, alpha_to_be_used, delta);

yPred_testsp_upper = permute(yPred_testsp_upper, [1 3 2]);
yPred_testsp_lower = permute(yPred_testsp_lower, [1 3 2]);

PI_testsp_post_values = [];
PINAW_testsp_post_values = [];

for i = 1:size(yPred_testsp_lower, 1)
    PI_testsp_post = PICP(yTrue_test, yPred_testsp_lower(i, :), yPred_testsp_upper(i, :));
    PINAW_testsp_post = PINAW(yTrue_test, yPred_testsp_lower(i,:), yPred_testsp_upper(i, :));
    PI_testsp_post_values = [PI_testsp_post, PI_testsp_post_values];
    PINAW_testsp_post_values = [PINAW_testsp_post, PINAW_testsp_post_values];
end

%%
function [X0, targets]  = create_mini_batch(X, yTrue, minibatch_size)

shuffle_idx = randperm(size(X, 3), minibatch_size);

X0 = X(:, :, shuffle_idx);
targets = yTrue(:, :, shuffle_idx);

if canUseGPU
    X0 = gpuArray(X0);
    targets = gpuArray(targets);
end

end

%%
function [mini_batch_inputs, targets] = call_batch(batch_inputs, batch_targets,iter,mbs)

mini_batch_inputs = batch_inputs(:, :, ((iter-1)*mbs)+1:(iter*mbs));
targets = batch_targets(:, :, ((iter-1)*mbs)+1:(iter*mbs));


end
