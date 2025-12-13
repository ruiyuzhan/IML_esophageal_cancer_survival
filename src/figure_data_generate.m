% =========================================================================
% File Description: Visualize data generation results by projecting 
%                   high-dimensional data to 2D space using PCA, showing the
%                   distribution of original and synthetic data.
% Copyright (c) [YEAR] [AUTHOR]. All rights reserved.
% =========================================================================

function figure_data_generate(origin_data, SyntheticData, origin_data_label, Synthetic_label)
% Plot generated data samples
% Input:
%   origin_data - Original feature data
%   SyntheticData - Generated synthetic data
%   origin_data_label - Original labels
%   Synthetic_label - Synthetic labels

% Check input data
if isempty(SyntheticData) || size(SyntheticData, 1) == 0
    fprintf('Warning: No synthetic data generated, skipping visualization\n');
    return;
end

% Check data dimensions
num_features_origin = size(origin_data, 2);
num_features_synth = size(SyntheticData, 2);

if num_features_origin ~= num_features_synth
    fprintf('Warning: Feature numbers of original and synthetic data do not match, skipping visualization\n');
    return;
end

% If data dimension is too high, use PCA to reduce to 2D for visualization
if num_features_origin > 2
    try
        % Combine data
        all_data = [origin_data; SyntheticData];
        
        % Check if there are enough samples for PCA (at least 2 samples needed)
        if size(all_data, 1) < 2
            fprintf('Warning: Insufficient data samples, skipping visualization\n');
            return;
        end
        
        % PCA dimensionality reduction
        [coeff, ~, ~] = pca(all_data);
        if size(coeff, 2) >= 2
            origin_data_2d = origin_data * coeff(:, 1:2);
            SyntheticData_2d = SyntheticData * coeff(:, 1:2);
        else
            % If PCA dimensions are insufficient, use first two features
            origin_data_2d = origin_data(:, 1:min(2, num_features_origin));
            SyntheticData_2d = SyntheticData(:, 1:min(2, num_features_synth));
        end
    catch ME
        % If PCA fails, use first two features
        fprintf('PCA dimensionality reduction failed, using first two features for visualization: %s\n', ME.message);
        if num_features_origin >= 2
            origin_data_2d = origin_data(:, 1:2);
            SyntheticData_2d = SyntheticData(:, 1:2);
        else
            fprintf('Warning: Less than 2 features, cannot visualize\n');
            return;
        end
    end
elseif num_features_origin == 2
    origin_data_2d = origin_data;
    SyntheticData_2d = SyntheticData;
else
    % Less than 2 features, cannot visualize
    fprintf('Warning: Less than 2 features, cannot perform 2D visualization\n');
    return;
end

% Create figure
figure('Name', 'Data Generation Visualization', 'Position', [100, 100, 800, 600]);

% Get unique labels
unique_labels = unique([origin_data_label; Synthetic_label]);
colors = lines(length(unique_labels));

% Plot original data
hold on;
for i = 1:length(unique_labels)
    label = unique_labels(i);
    orig_idx = origin_data_label == label;
    if any(orig_idx)
        scatter(origin_data_2d(orig_idx, 1), origin_data_2d(orig_idx, 2), ...
               50, colors(i, :), 'o', 'filled', 'DisplayName', sprintf('Original-Class%d', label));
    end
end

% Plot synthetic data
for i = 1:length(unique_labels)
    label = unique_labels(i);
    synth_idx = Synthetic_label == label;
    if any(synth_idx)
        scatter(SyntheticData_2d(synth_idx, 1), SyntheticData_2d(synth_idx, 2), ...
               30, colors(i, :), 'x', 'LineWidth', 1.5, 'DisplayName', sprintf('Synthetic-Class%d', label));
    end
end

hold off;
xlabel('Feature 1 (PCA)');
ylabel('Feature 2 (PCA)');
title('Data Generation Visualization');
legend('Location', 'best');
grid on;

fprintf('Data generation visualization completed\n');
fprintf('Original data: %d samples\n', size(origin_data, 1));
fprintf('Synthetic data: %d samples\n', size(SyntheticData, 1));
fprintf('Total: %d samples\n', size(origin_data, 1) + size(SyntheticData, 1));

end

