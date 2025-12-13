% =========================================================================
% File Description: Generate synthetic data for data augmentation. Supports
%                   three methods: SMOTE-like, simple copy, and adding noise,
%                   for balancing datasets and increasing training sample size.
% Copyright (c) [YEAR] [AUTHOR]. All rights reserved.
% =========================================================================

function [SyntheticData, Synthetic_label] = generate_classdata(origin_data, origin_data_label, methodchoose, get_mutiple)
% Generate synthetic data function
% Input:
%   origin_data - Original feature data (N x M)
%   origin_data_label - Original labels (N x 1)
%   methodchoose - Method selection (1=SMOTE-like, 2=simple copy, 3=add noise)
%   get_mutiple - Data augmentation multiplier
% Output:
%   SyntheticData - Generated synthetic feature data
%   Synthetic_label - Generated synthetic labels

if nargin < 3
    methodchoose = 1;
end
if nargin < 4
    get_mutiple = 1;
end

% Check input data
if isempty(origin_data) || size(origin_data, 1) == 0
    SyntheticData = [];
    Synthetic_label = [];
    fprintf('Warning: Original data is empty, cannot generate synthetic data\n');
    return;
end

% Calculate number of samples to generate
original_size = size(origin_data, 1);
num_features = size(origin_data, 2);

if get_mutiple <= 1
    % If multiplier <= 1, do not generate new data
    SyntheticData = [];
    Synthetic_label = [];
    fprintf('Data augmentation multiplier <= 1, not generating synthetic data\n');
    return;
end

target_size = round(original_size * get_mutiple);
num_to_generate = target_size - original_size;

if num_to_generate <= 0
    % If no new data needs to be generated, return empty
    SyntheticData = [];
    Synthetic_label = [];
    fprintf('No new data needs to be generated\n');
    return;
end

fprintf('Starting synthetic data generation: Original samples=%d, Target samples=%d, To generate=%d\n', ...
        original_size, target_size, num_to_generate);

switch methodchoose
    case 1
        % Method 1: SMOTE-like method (interpolation between samples of the same class)
        [SyntheticData, Synthetic_label] = generate_smote_like(origin_data, origin_data_label, num_to_generate);
        
    case 2
        % Method 2: Simple copy (with small random perturbation)
        [SyntheticData, Synthetic_label] = generate_simple_copy(origin_data, origin_data_label, num_to_generate);
        
    case 3
        % Method 3: Add Gaussian noise
        [SyntheticData, Synthetic_label] = generate_with_noise(origin_data, origin_data_label, num_to_generate);
        
    otherwise
        % Default: use method 1
        [SyntheticData, Synthetic_label] = generate_smote_like(origin_data, origin_data_label, num_to_generate);
end

end

% SMOTE-like method
function [SyntheticData, Synthetic_label] = generate_smote_like(origin_data, origin_data_label, num_to_generate)
    % Check input
    if isempty(origin_data) || num_to_generate <= 0
        SyntheticData = [];
        Synthetic_label = [];
        return;
    end
    
    num_features = size(origin_data, 2);
    
    % Get unique labels
    unique_labels = unique(origin_data_label);
    num_labels = length(unique_labels);
    
    % Generate samples for each class
    samples_per_label = ceil(num_to_generate / num_labels);
    SyntheticData = zeros(0, num_features);  % Initialize with correct dimensions
    Synthetic_label = [];
    
    for i = 1:num_labels
        label = unique_labels(i);
        % Find all samples with this label
        label_indices = find(origin_data_label == label);
        label_data = origin_data(label_indices, :);
        num_label_samples = length(label_indices);
        
        if num_label_samples < 2
            % If this class has fewer than 2 samples, use simple copy
            num_to_gen = min(samples_per_label, num_to_generate - size(SyntheticData, 1));
            if num_to_gen > 0
                synth = repmat(label_data, ceil(num_to_gen / num_label_samples), 1);
                synth = synth(1:num_to_gen, :);
                % Add small amount of noise
                noise = randn(size(synth)) * 0.01 * std(label_data, 1);
                synth = synth + noise;
                SyntheticData = [SyntheticData; synth];
                Synthetic_label = [Synthetic_label; repmat(label, num_to_gen, 1)];
            end
        else
            % Interpolate between samples of the same class
            num_to_gen = min(samples_per_label, num_to_generate - size(SyntheticData, 1));
            synth = [];
            for j = 1:num_to_gen
                % Randomly select two samples
                idx1 = label_indices(randi(num_label_samples));
                idx2 = label_indices(randi(num_label_samples));
                while idx2 == idx1 && num_label_samples > 1
                    idx2 = label_indices(randi(num_label_samples));
                end
                
                % Interpolate between two points
                alpha = rand();
                new_sample = origin_data(idx1, :) * (1 - alpha) + origin_data(idx2, :) * alpha;
                synth = [synth; new_sample];
            end
            SyntheticData = [SyntheticData; synth];
            Synthetic_label = [Synthetic_label; repmat(label, num_to_gen, 1)];
        end
    end
    
    % If generated samples exceed requirement, randomly select
    if size(SyntheticData, 1) > num_to_generate
        indices = randperm(size(SyntheticData, 1), num_to_generate);
        SyntheticData = SyntheticData(indices, :);
        Synthetic_label = Synthetic_label(indices, :);
    end
end

% Simple copy method
function [SyntheticData, Synthetic_label] = generate_simple_copy(origin_data, origin_data_label, num_to_generate)
    % Check input
    if isempty(origin_data) || num_to_generate <= 0
        SyntheticData = [];
        Synthetic_label = [];
        return;
    end
    
    % Randomly select samples and add small amount of noise
    indices = randi(size(origin_data, 1), num_to_generate, 1);
    SyntheticData = origin_data(indices, :);
    Synthetic_label = origin_data_label(indices, :);
    
    % Add small random noise
    noise_scale = 0.01;
    std_data = std(origin_data, 0, 1);  % Calculate standard deviation by column
    std_data(std_data == 0) = 1;  % Avoid division by zero
    noise = randn(size(SyntheticData)) .* repmat(std_data * noise_scale, size(SyntheticData, 1), 1);
    SyntheticData = SyntheticData + noise;
end

% Add noise method
function [SyntheticData, Synthetic_label] = generate_with_noise(origin_data, origin_data_label, num_to_generate)
    % Check input
    if isempty(origin_data) || num_to_generate <= 0
        SyntheticData = [];
        Synthetic_label = [];
        return;
    end
    
    % Randomly select samples
    indices = randi(size(origin_data, 1), num_to_generate, 1);
    SyntheticData = origin_data(indices, :);
    Synthetic_label = origin_data_label(indices, :);
    
    % Add Gaussian noise
    noise_scale = 0.05;
    std_data = std(origin_data, 0, 1);  % Calculate standard deviation by column
    std_data(std_data == 0) = 1;  % Avoid division by zero
    noise = randn(size(SyntheticData)) .* repmat(std_data * noise_scale, size(SyntheticData, 1), 1);
    SyntheticData = SyntheticData + noise;
end

