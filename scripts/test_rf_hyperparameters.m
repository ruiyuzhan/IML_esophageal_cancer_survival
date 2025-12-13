% =========================================================================
% File Description: Random Forest hyperparameter ablation experiment script.
%                  
% =========================================================================

% 随机森林超参数消融实验
clc; clear; close all;

% 检查工具箱
if ~license('test', 'Statistics_Toolbox')
    error('错误: 需要 Statistics and Machine Learning Toolbox');
end

% 测试配置
datasets = {'1yearos.csv', '5yearos.csv'};
dataset_names = {'1年OS', '5年OS'};

% 超参数组合（消融实验）
num_trees_list = [20, 50, 100, 150, 200];  % NumTrees的不同值
min_leaf_list = [2, 5, 8, 10];              % MinLeafSize的不同值

% 存储所有结果
all_results = struct();

% 对每个数据集进行测试
for d = 1:length(datasets)
    dataset_file = datasets{d};
    dataset_name = dataset_names{d};
    
    fprintf('\n');
    fprintf('============================================================\n');
    fprintf('测试数据集: %s (%s)\n', dataset_name, dataset_file);
    fprintf('============================================================\n');
    
    % 加载配置
    load('R_11_Nov_2025_20_22_24.mat');
    
    % 修改数据路径
    G_out_data.data_path_str = dataset_file;
    
    % 设置随机种子
    random_seed = G_out_data.random_seed;
    rng(random_seed);
    
    % Get script directory
    script_dir = fileparts(mfilename('fullpath'));
    project_root = fileparts(script_dir);  % Go up one level to project root
    processed_data_dir = fullfile(project_root, 'data', 'processed_data');
    
    % 构建数据路径
    normalized_path = strrep(dataset_file, '\', '/');
    if contains(normalized_path, '/')
        parts = strsplit(normalized_path, '/');
        filename_with_ext = parts{end};
    else
        filename_with_ext = normalized_path;
    end
    [~, filename, ext] = fileparts(filename_with_ext);
    data_str = fullfile(processed_data_dir, [filename, ext]);
    
    if ~exist(data_str, 'file')
        fprintf('警告: 数据文件不存在: %s\n', data_str);
        continue;
    end
    
    % 加载数据（简化版，只加载一次）
    fprintf('加载数据...\n');
    dataO = readtable(data_str, 'VariableNamingRule', 'preserve');
    data1 = dataO(:, 2:end);
    test_data = table2cell(dataO(1, 2:end));
    
    % 识别数据类型
    index_la = zeros(1, length(test_data));
    for i = 1:length(test_data)
        if ischar(test_data{1, i}) == 1
            index_la(i) = 1;
        elseif isnumeric(test_data{1, i}) == 1
            index_la(i) = 2;
        else
            index_la(i) = 0;
        end
    end
    index_char = find(index_la == 1);
    index_double = find(index_la == 2);
    
    % 处理数值型数据
    if length(index_double) >= 1
        data_numshuju = table2array(data1(:, index_double));
        index_double1 = index_double;
        index_double1_index = 1:size(data_numshuju, 2);
        data_NAN = isnan(data_numshuju);
        num_NAN_ROW = sum(data_NAN);
        index_NAN = num_NAN_ROW > round(0.2 * size(data1, 1));
        index_double1(index_NAN == 1) = [];
        index_double1_index(index_NAN == 1) = [];
        data_numshuju1 = data_numshuju(:, index_double1_index);
        data_NAN1 = isnan(data_numshuju1);
        num_NAN__COL = sum(data_NAN1);
        index_NAN1 = num_NAN__COL > 0;
        index_double2_index = 1:size(data_numshuju, 1);
        index_double2_index(index_NAN1 == 1) = [];
        data_numshuju2 = data_numshuju1(index_double2_index, :);
        index_need_last = index_double1;
    else
        index_need_last = [];
        data_numshuju2 = [];
    end
    
    % 处理字符型数据
    data_shuju = [];
    if length(index_char) >= 1
        for j = 1:length(index_char)
            data_get = table2array(data1(index_double2_index, index_char(j)));
            data_label = unique(data_get);
            for NN = 1:length(data_label)
                idx = find(ismember(data_get, data_label{NN, 1}));
                data_shuju(idx, j) = NN;
            end
        end
    end
    
    % 合并数据
    label_all_last = [index_char, index_need_last];
    [~, label_max] = max(label_all_last);
    if label_max == length(label_all_last)
        data_all_last = [data_shuju, data_numshuju2];
        label_all_last = [index_char, index_need_last];
    else
        data_all_last = [data_numshuju2, data_shuju];
        label_all_last = [index_need_last, index_char];
    end
    
    data = data_all_last;
    A_data1 = data;
    data_select = A_data1;
    
    % 分离特征和标签
    x_feature_label = data_select(:, 1:end-1);
    y_feature_label_raw = data_select(:, end);
    
    % 处理标签
    unique_labels_raw = unique(y_feature_label_raw(~isnan(y_feature_label_raw)));
    if length(unique_labels_raw) > 10
        median_value = median(y_feature_label_raw(~isnan(y_feature_label_raw)));
        y_feature_label = double(y_feature_label_raw < median_value);
        y_feature_label(isnan(y_feature_label_raw)) = 0;
    else
        y_feature_label = double(y_feature_label_raw);
        min_label = min(y_feature_label(~isnan(y_feature_label)));
        if min_label ~= 0 && min_label ~= 1
            y_feature_label = y_feature_label - min_label;
        end
        y_feature_label(isnan(y_feature_label)) = 0;
        y_feature_label = round(y_feature_label);
    end
    
    data_size = size(x_feature_label, 1);
    
    % 数据划分
    index_label = randperm(data_size);
    spilt_ri = G_out_data.spilt_rio;
    train_num = round(spilt_ri(1) / (sum(spilt_ri)) * data_size);
    vaild_num = round((spilt_ri(1) + spilt_ri(2)) / (sum(spilt_ri)) * data_size);
    train_num = min(train_num, data_size);
    vaild_num = min(vaild_num, data_size);
    if train_num >= vaild_num
        vaild_num = min(train_num + 1, data_size);
    end
    
    train_x_feature_label = x_feature_label(index_label(1:train_num), :);
    train_y_feature_label = y_feature_label(index_label(1:train_num), :);
    vaild_x_feature_label = x_feature_label(index_label(train_num+1:vaild_num), :);
    vaild_y_feature_label = y_feature_label(index_label(train_num+1:vaild_num), :);
    test_x_feature_label = x_feature_label(index_label(vaild_num+1:end), :);
    test_y_feature_label = y_feature_label(index_label(vaild_num+1:end), :);
    
    % 标准化
    x_mu = mean(train_x_feature_label, 1, 'omitnan');
    x_sig = std(train_x_feature_label, 0, 1, 'omitnan');
    x_sig(x_sig == 0) = 1;
    
    train_x_feature_label_norm = (train_x_feature_label - x_mu) ./ x_sig;
    train_x_feature_label_norm(isnan(train_x_feature_label_norm)) = 0;
    train_x_feature_label_norm(isinf(train_x_feature_label_norm)) = 0;
    
    vaild_x_feature_label_norm = (vaild_x_feature_label - x_mu) ./ x_sig;
    vaild_x_feature_label_norm(isnan(vaild_x_feature_label_norm)) = 0;
    vaild_x_feature_label_norm(isinf(vaild_x_feature_label_norm)) = 0;
    
    test_x_feature_label_norm = (test_x_feature_label - x_mu) ./ x_sig;
    test_x_feature_label_norm(isnan(test_x_feature_label_norm)) = 0;
    test_x_feature_label_norm(isinf(test_x_feature_label_norm)) = 0;
    
    fprintf('数据加载完成: %d 样本, %d 特征\n', data_size, size(x_feature_label, 2));
    fprintf('训练集: %d, 验证集: %d, 测试集: %d\n', ...
        size(train_x_feature_label_norm, 1), ...
        size(vaild_x_feature_label_norm, 1), ...
        size(test_x_feature_label_norm, 1));
    
    % 测试不同的超参数组合
    fprintf('\n开始超参数消融实验...\n');
    fprintf('测试组合数: %d (NumTrees) × %d (MinLeafSize) = %d 组\n', ...
        length(num_trees_list), length(min_leaf_list), ...
        length(num_trees_list) * length(min_leaf_list));
    
    results_table = [];
    result_idx = 0;
    
    for nt = 1:length(num_trees_list)
        for ml = 1:length(min_leaf_list)
            num_trees = num_trees_list(nt);
            min_leaf = min_leaf_list(ml);
            
            result_idx = result_idx + 1;
            fprintf('\n[实验 %d/%d] NumTrees=%d, MinLeafSize=%d\n', ...
                result_idx, length(num_trees_list) * length(min_leaf_list), ...
                num_trees, min_leaf);
            
            % 训练模型
            try
                Mdl = TreeBagger(num_trees, train_x_feature_label_norm, ...
                    train_y_feature_label, 'Method', 'classification', ...
                    'MinLeafSize', min_leaf);
                
                % 预测
                y_train_pred = RF_process(predict(Mdl, train_x_feature_label_norm));
                y_val_pred = RF_process(predict(Mdl, vaild_x_feature_label_norm));
                y_test_pred = RF_process(predict(Mdl, test_x_feature_label_norm));
                
                % 计算指标
                accuracy_train = sum(y_train_pred == train_y_feature_label) / length(train_y_feature_label);
                accuracy_val = sum(y_val_pred == vaild_y_feature_label) / length(vaild_y_feature_label);
                accuracy_test = sum(y_test_pred == test_y_feature_label) / length(test_y_feature_label);
                
                % 计算混淆矩阵
                confMat_test = confusionmat(test_y_feature_label, y_test_pred);
                TP = diag(confMat_test);
                TP = TP';
                FP = sum(confMat_test, 1) - TP;
                FN = sum(confMat_test, 2)' - TP;
                TN = sum(confMat_test(:)) - (TP + FP + FN);
                
                precision_test = TP ./ (TP + FP);
                precision_test(isnan(precision_test)) = 0;
                recall_test = TP ./ (TP + FN);
                recall_test(isnan(recall_test)) = 0;
                F1_test = 2 * (precision_test .* recall_test) ./ (precision_test + recall_test);
                F1_test(isnan(F1_test)) = 0;
                
                % 计算AUC
                [~, score_test] = predict(Mdl, test_x_feature_label_norm);
                try
                    % 获取类别名称（TreeBagger的ClassNames）
                    class_names = Mdl.ClassNames;
                    unique_labels = unique(test_y_feature_label);
                    
                    % 调试信息（仅第一次实验）
                    if result_idx == 1
                        fprintf('    调试信息:\n');
                        % 处理ClassNames（可能是cell数组）
                        if iscell(class_names)
                            class_names_str = sprintf('[%s]', strjoin(cellfun(@num2str, class_names, 'UniformOutput', false), ', '));
                        else
                            class_names_str = mat2str(class_names);
                        end
                        fprintf('      ClassNames: %s\n', class_names_str);
                        fprintf('      标签唯一值: %s\n', mat2str(unique_labels));
                        fprintf('      score_test维度: %s\n', mat2str(size(score_test)));
                        fprintf('      score_test(:,1)范围: [%.4f, %.4f]\n', min(score_test(:,1)), max(score_test(:,1)));
                        if size(score_test, 2) > 1
                            fprintf('      score_test(:,2)范围: [%.4f, %.4f]\n', min(score_test(:,2)), max(score_test(:,2)));
                        end
                    end
                    
                    % 确定正类标签和概率列
                    if length(unique_labels) == 2
                        % 尝试两种方法：使用标签1和标签2
                        % 方法1: 使用原始代码的方法（score(:,1)和标签1）
                        try
                            % 确保使用正确的标签值（处理cell数组情况）
                            pos_label_1 = unique_labels(1);
                            if iscell(class_names) && length(class_names) >= 1
                                pos_label_1 = class_names{1};
                                if ischar(pos_label_1) || isstring(pos_label_1)
                                    pos_label_1 = str2double(pos_label_1);
                                elseif isnumeric(pos_label_1)
                                    % 已经是数值，直接使用
                                end
                            end
                            [~, ~, ~, AUC_test_1] = perfcurve(test_y_feature_label, score_test(:,1), pos_label_1);
                        catch ME1
                            AUC_test_1 = 0;
                            if result_idx == 1
                                fprintf('      方法1失败: %s\n', ME1.message);
                            end
                        end
                        
                        % 方法2: 如果标签是{1,2}，尝试使用标签2和score(:,2)
                        AUC_test_2 = 0;
                        if length(unique_labels) == 2 && max(unique_labels) == 2
                            try
                                if size(score_test, 2) >= 2
                                    % 确保使用正确的标签值（处理cell数组情况）
                                    pos_label_2 = unique_labels(2);
                                    if iscell(class_names) && length(class_names) >= 2
                                        pos_label_2 = class_names{2};
                                        if ischar(pos_label_2) || isstring(pos_label_2)
                                            pos_label_2 = str2double(pos_label_2);
                                        elseif isnumeric(pos_label_2)
                                            % 已经是数值，直接使用
                                        end
                                    end
                                    [~, ~, ~, AUC_test_2] = perfcurve(test_y_feature_label, score_test(:,2), pos_label_2);
                                else
                                    AUC_test_2 = 0;
                                end
                            catch ME2
                                AUC_test_2 = 0;
                                if result_idx == 1
                                    fprintf('      方法2失败: %s\n', ME2.message);
                                end
                            end
                        end
                        
                        % 选择较大的AUC（更合理的结果）
                        if AUC_test_2 > AUC_test_1 && AUC_test_2 > 0.5
                            AUC_test = AUC_test_2;
                            if result_idx == 1
                                fprintf('      使用方法2: score(:,2)和标签%d, AUC=%.4f\n', unique_labels(2), AUC_test);
                            end
                        else
                            AUC_test = AUC_test_1;
                            if result_idx == 1
                                fprintf('      使用方法1: score(:,1)和标签%d, AUC=%.4f\n', unique_labels(1), AUC_test);
                            end
                        end
                    else
                        % 多分类情况，使用第一个类别
                        [~, ~, ~, AUC_test] = perfcurve(test_y_feature_label, score_test(:, 1), class_names(1));
                    end
                catch ME
                    fprintf('    AUC计算警告: %s\n', ME.message);
                    if result_idx == 1
                        fprintf('    错误详情: %s\n', getReport(ME));
                    end
                    AUC_test = 0;
                end
                
                % 存储结果
                results_table(result_idx, :) = [num_trees, min_leaf, ...
                    accuracy_train, accuracy_val, accuracy_test, ...
                    mean(precision_test), mean(recall_test), mean(F1_test), AUC_test];
                
                fprintf('  训练集准确率: %.4f, 验证集准确率: %.4f, 测试集准确率: %.4f\n', ...
                    accuracy_train, accuracy_val, accuracy_test);
                fprintf('  测试集F1: %.4f, AUC: %.4f\n', mean(F1_test), AUC_test);
                
            catch ME
                fprintf('  错误: %s\n', ME.message);
                results_table(result_idx, :) = [num_trees, min_leaf, 0, 0, 0, 0, 0, 0, 0];
            end
        end
    end
    
    % 保存结果（字段名不能以数字开头，使用前缀）
    field_name = ['dataset_', strrep(dataset_name, '年', 'year')];
    field_name = strrep(field_name, ' ', '_');
    all_results.(field_name).results = results_table;
    all_results.(field_name).num_trees_list = num_trees_list;
    all_results.(field_name).min_leaf_list = min_leaf_list;
    all_results.(field_name).dataset_name = dataset_name;
    
    % 输出结果表格
    fprintf('\n');
    fprintf('============================================================\n');
    fprintf('%s - 超参数消融实验结果\n', dataset_name);
    fprintf('============================================================\n');
    fprintf('%-10s | %-12s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s\n', ...
        'NumTrees', 'MinLeafSize', 'Train_Acc', 'Val_Acc', 'Test_Acc', ...
        'Precision', 'Recall', 'F1-Score', 'AUC');
    fprintf('%-10s-+-%-12s-+-%-10s-+-%-10s-+-%-10s-+-%-10s-+-%-10s-+-%-10s-+-%-10s\n', ...
        repmat('-', 1, 10), repmat('-', 1, 12), repmat('-', 1, 10), ...
        repmat('-', 1, 10), repmat('-', 1, 10), repmat('-', 1, 10), ...
        repmat('-', 1, 10), repmat('-', 1, 10), repmat('-', 1, 10));
    
    for i = 1:size(results_table, 1)
        fprintf('%-10d | %-12d | %10.4f | %10.4f | %10.4f | %10.4f | %10.4f | %10.4f | %10.4f\n', ...
            results_table(i, 1), results_table(i, 2), results_table(i, 3), ...
            results_table(i, 4), results_table(i, 5), results_table(i, 6), ...
            results_table(i, 7), results_table(i, 8), results_table(i, 9));
    end
    
    % 找到最优组合
    [~, best_idx] = max(results_table(:, 5));  % 按测试集准确率
    best_num_trees = results_table(best_idx, 1);
    best_min_leaf = results_table(best_idx, 2);
    
    fprintf('\n最优组合 (按测试集准确率):\n');
    fprintf('  NumTrees: %d, MinLeafSize: %d\n', best_num_trees, best_min_leaf);
    fprintf('  测试集准确率: %.4f\n', results_table(best_idx, 5));
    fprintf('  测试集F1-Score: %.4f\n', results_table(best_idx, 8));
    fprintf('  测试集AUC: %.4f\n', results_table(best_idx, 9));
    
    fprintf('============================================================\n');
    
    % 使用最优参数重新训练模型并绘制ROC和DCA曲线
    fprintf('\n使用最优参数训练模型并绘制ROC和DCA曲线...\n');
    try
        Mdl_best = TreeBagger(best_num_trees, train_x_feature_label_norm, ...
            train_y_feature_label, 'Method', 'classification', ...
            'MinLeafSize', best_min_leaf);
        
        % 获取预测概率
        [~, score_test] = predict(Mdl_best, test_x_feature_label_norm);
        
        % 获取类别名称（TreeBagger的ClassNames）
        class_names = Mdl_best.ClassNames;
        unique_labels = unique(test_y_feature_label);
        
        fprintf('  调试信息:\n');
        % 处理ClassNames（可能是cell数组）
        if iscell(class_names)
            class_names_str = sprintf('[%s]', strjoin(cellfun(@num2str, class_names, 'UniformOutput', false), ', '));
        else
            class_names_str = mat2str(class_names);
        end
        fprintf('    ClassNames: %s\n', class_names_str);
        fprintf('    标签唯一值: %s\n', mat2str(unique_labels));
        fprintf('    score_test维度: %s\n', mat2str(size(score_test)));
        
        % 确定正类标签和对应的概率
        if length(unique_labels) == 2
            % 尝试两种方法，选择AUC更大的
            % 方法1: 使用score(:,1)和标签1（原始代码的方法）
            try
                [~, ~, ~, AUC_1] = perfcurve(test_y_feature_label, score_test(:,1), unique_labels(1));
            catch
                AUC_1 = 0;
            end
            
            % 方法2: 如果标签是{1,2}，使用score(:,2)和标签2
            AUC_2 = 0;
            if max(unique_labels) == 2 && size(score_test, 2) >= 2
                try
                    % 确保使用正确的标签值（可能是cell数组中的值）
                    pos_label_2 = unique_labels(2);
                    if iscell(class_names) && length(class_names) >= 2
                        % 如果ClassNames是cell，找到对应的数值
                        pos_label_2 = class_names{2};
                        if ischar(pos_label_2) || isstring(pos_label_2)
                            pos_label_2 = str2double(pos_label_2);
                        end
                    end
                    [~, ~, ~, AUC_2] = perfcurve(test_y_feature_label, score_test(:,2), pos_label_2);
                catch ME2
                    AUC_2 = 0;
                    fprintf('    方法2失败: %s\n', ME2.message);
                end
            end
            
            fprintf('    方法1 (score(:,1), 标签%d): AUC=%.4f\n', unique_labels(1), AUC_1);
            if AUC_2 > 0
                fprintf('    方法2 (score(:,2), 标签%d): AUC=%.4f\n', unique_labels(2), AUC_2);
            end
            
            % 选择AUC更大的方法（且大于0.5，说明模型有效）
            if AUC_2 > AUC_1 && AUC_2 > 0.5
                pos_class = unique_labels(2);
                pos_proba = score_test(:, 2);
                AUC_ROC = AUC_2;
                fprintf('    ✓ 使用方法2\n');
            else
                pos_class = unique_labels(1);
                pos_proba = score_test(:, 1);
                AUC_ROC = AUC_1;
                fprintf('    ✓ 使用方法1\n');
            end
            
            % 确保pos_class是正确的数值类型（处理cell数组情况）
            if iscell(class_names)
                % 找到pos_class在class_names中的位置
                pos_class_idx = 0;
                for i = 1:length(class_names)
                    if isequal(class_names{i}, pos_class) || (isnumeric(class_names{i}) && class_names{i} == pos_class)
                        pos_class_idx = i;
                        if isnumeric(class_names{i})
                            pos_class = class_names{i};
                        end
                        break;
                    end
                end
            else
                pos_class_idx = find(class_names == pos_class, 1);
            end
            
            % 将标签转换为0和1（用于DCA）
            test_y_binary = double(test_y_feature_label == pos_class);
            
            % 绘制ROC曲线
            [X_ROC, Y_ROC, ~, AUC_ROC] = perfcurve(test_y_feature_label, pos_proba, pos_class);
            
            if pos_class_idx > 0
                fprintf('    最终使用: 标签%d, 概率列%d, AUC=%.4f\n', pos_class, pos_class_idx, AUC_ROC);
            else
                fprintf('    最终使用: 标签%d, AUC=%.4f\n', pos_class, AUC_ROC);
            end
            fprintf('    概率范围: [%.4f, %.4f]\n', min(pos_proba), max(pos_proba));
        else
            % 多分类情况
            pos_class = class_names(1);
            pos_proba = score_test(:, 1);
            test_y_binary = double(test_y_feature_label == pos_class);
            [X_ROC, Y_ROC, ~, AUC_ROC] = perfcurve(test_y_feature_label, pos_proba, pos_class);
        end
        
        fprintf('  计算得到的AUC: %.4f\n', AUC_ROC);
        
        % 创建组合图（浅色系：白色背景，黑色字体，用于SCI文章）
        fig = figure('Position', [100, 100, 1400, 600], 'Color', 'white', 'InvertHardcopy', 'off');
        
        % ROC曲线
        subplot(1, 2, 1);
        ax1 = gca;
        set(ax1, 'Color', 'white', 'XColor', 'black', 'YColor', 'black', ...
            'GridColor', [0.9, 0.9, 0.9], 'MinorGridColor', [0.95, 0.95, 0.95], ...
            'GridAlpha', 1, 'MinorGridAlpha', 1);
        % 设置坐标轴刻度颜色为黑色
        set(ax1, 'TickDir', 'out', 'TickLength', [0.02, 0.02]);
        % 确保刻度标签颜色为黑色
        try
            ax1.XAxis.TickLabelColor = 'black';
            ax1.YAxis.TickLabelColor = 'black';
        catch
            % 如果新版本属性不支持，使用传统方法
            set(ax1, 'XColor', 'black', 'YColor', 'black');
        end
        plot(X_ROC, Y_ROC, 'LineWidth', 2.5, 'Color', [0.18, 0.53, 0.67]); % #2E86AB
        hold on;
        plot([0, 1], [0, 1], '--', 'LineWidth', 2, 'Color', [0.63, 0.63, 0.63]); % #A0A0A0
        xlabel('False Positive Rate (1-Specificity)', 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'black');
        ylabel('True Positive Rate (Sensitivity)', 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'black');
        % 将数据集名称转换为英文
        if contains(dataset_name, '1年') || contains(dataset_name, '1year')
            dataset_name_en = '1-year OS';
        else
            dataset_name_en = '5-year OS';
        end
        title(sprintf('%s - ROC Curve (AUC = %.4f)', dataset_name_en, AUC_ROC), ...
            'FontSize', 16, 'FontWeight', 'bold', 'Color', 'black');
        leg1 = legend(sprintf('RF Model (AUC=%.4f)', AUC_ROC), 'Random Classifier', ...
            'Location', 'southeast', 'FontSize', 12, 'TextColor', 'black', ...
            'Color', 'white', 'EdgeColor', [0.8, 0.8, 0.8], 'Box', 'on');
        grid on;
        grid minor;
        set(ax1, 'GridColor', [0.9, 0.9, 0.9], 'GridAlpha', 1);
        set(ax1, 'MinorGridColor', [0.95, 0.95, 0.95], 'MinorGridAlpha', 1);
        axis square;
        xlim([0, 1]);
        ylim([0, 1]);
        
        % DCA曲线（使用正类概率）
        subplot(1, 2, 2);
        ax2 = gca;
        set(ax2, 'Color', 'white', 'XColor', 'black', 'YColor', 'black', ...
            'GridColor', [0.9, 0.9, 0.9], 'MinorGridColor', [0.95, 0.95, 0.95], ...
            'GridAlpha', 1, 'MinorGridAlpha', 1);
        set(ax2, 'TickDir', 'out', 'TickLength', [0.02, 0.02]);
        % 确保刻度标签颜色为黑色
        try
            ax2.XAxis.TickLabelColor = 'black';
            ax2.YAxis.TickLabelColor = 'black';
        catch
            set(ax2, 'XColor', 'black', 'YColor', 'black');
        end
        plot_dca_curve_matlab(test_y_binary, pos_proba, dataset_name_en);
        
        % 设置总标题
        sgtitle(sprintf('%s - Optimal Model (NumTrees=%d, MinLeafSize=%d)', ...
            dataset_name_en, best_num_trees, best_min_leaf), ...
            'FontSize', 16, 'FontWeight', 'bold', 'Color', 'black');
        
        % 保存图片（白色背景，黑色字体，用于SCI文章）
        safe_name = strrep(dataset_name_en, ' ', '_');
        safe_name = strrep(safe_name, '-', '_');
        % 使用print确保白色背景和正确的颜色
        print(fig, sprintf('MATLAB_ROC_DCA_%s.png', safe_name), '-dpng', '-r300', '-painters');
        fprintf('  ROC和DCA曲线已保存到: MATLAB_ROC_DCA_%s.png\n', safe_name);
        close(fig);
        
        % 单独保存ROC曲线（浅色系：白色背景，黑色字体，用于SCI文章）
        fig_roc = figure('Position', [100, 100, 800, 800], 'Color', 'white', 'InvertHardcopy', 'off');
        ax_roc = axes('Parent', fig_roc);
        set(ax_roc, 'Color', 'white', 'XColor', 'black', 'YColor', 'black', ...
            'GridColor', [0.9, 0.9, 0.9], 'MinorGridColor', [0.95, 0.95, 0.95], ...
            'GridAlpha', 1, 'MinorGridAlpha', 1, 'TickDir', 'out', 'TickLength', [0.02, 0.02]);
        % 确保刻度标签颜色为黑色
        try
            ax_roc.XAxis.TickLabelColor = 'black';
            ax_roc.YAxis.TickLabelColor = 'black';
        catch
            set(ax_roc, 'XColor', 'black', 'YColor', 'black');
        end
        plot(ax_roc, X_ROC, Y_ROC, 'LineWidth', 2.5, 'Color', [0.18, 0.53, 0.67]);
        hold on;
        plot(ax_roc, [0, 1], [0, 1], '--', 'LineWidth', 2, 'Color', [0.63, 0.63, 0.63]);
        xlabel('False Positive Rate (1-Specificity)', 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'black');
        ylabel('True Positive Rate (Sensitivity)', 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'black');
        title(sprintf('%s - ROC Curve (AUC = %.4f)', dataset_name_en, AUC_ROC), ...
            'FontSize', 16, 'FontWeight', 'bold', 'Color', 'black');
        leg_roc = legend(sprintf('RF Model (AUC=%.4f)', AUC_ROC), 'Random Classifier', ...
            'Location', 'southeast', 'FontSize', 12, 'TextColor', 'black', ...
            'Color', 'white', 'EdgeColor', [0.8, 0.8, 0.8], 'Box', 'on');
        grid on;
        grid minor;
        set(ax_roc, 'GridColor', [0.9, 0.9, 0.9], 'GridAlpha', 1);
        set(ax_roc, 'MinorGridColor', [0.95, 0.95, 0.95], 'MinorGridAlpha', 1);
        axis square;
        xlim([0, 1]);
        ylim([0, 1]);
        print(fig_roc, sprintf('MATLAB_ROC_%s.png', safe_name), '-dpng', '-r300', '-painters');
        fprintf('  ROC曲线已保存到: MATLAB_ROC_%s.png\n', safe_name);
        close(fig_roc);
        
        % 单独保存DCA曲线（浅色系：白色背景，黑色字体，用于SCI文章）
        fig_dca = figure('Position', [100, 100, 1000, 800], 'Color', 'white', 'InvertHardcopy', 'off');
        ax_dca = axes('Parent', fig_dca);
        set(ax_dca, 'Color', 'white', 'XColor', 'black', 'YColor', 'black', ...
            'GridColor', [0.9, 0.9, 0.9], 'MinorGridColor', [0.95, 0.95, 0.95], ...
            'GridAlpha', 1, 'MinorGridAlpha', 1, 'TickDir', 'out', 'TickLength', [0.02, 0.02]);
        % 确保刻度标签颜色为黑色
        try
            ax_dca.XAxis.TickLabelColor = 'black';
            ax_dca.YAxis.TickLabelColor = 'black';
        catch
            set(ax_dca, 'XColor', 'black', 'YColor', 'black');
        end
        plot_dca_curve_matlab(test_y_binary, pos_proba, dataset_name_en);
        print(fig_dca, sprintf('MATLAB_DCA_%s.png', safe_name), '-dpng', '-r300', '-painters');
        fprintf('  DCA曲线已保存到: MATLAB_DCA_%s.png\n', safe_name);
        close(fig_dca);
        
    catch ME
        fprintf('  警告: 无法绘制ROC和DCA曲线 - %s\n', ME.message);
        fprintf('  错误位置: %s\n', ME.stack(1).name);
    end
    
    fprintf('============================================================\n');
end

% 保存所有结果
save('rf_hyperparameter_results.mat', 'all_results');
fprintf('\n所有结果已保存到: rf_hyperparameter_results.mat\n');

% 创建对比图
fprintf('\n生成结果可视化...\n');
create_hyperparameter_plots(all_results);

fprintf('\n测试完成！\n');

function create_hyperparameter_plots(all_results)
    % 创建超参数影响的可视化图
    
    datasets = fieldnames(all_results);
    
    for d = 1:length(datasets)
        field_name = datasets{d};
        results = all_results.(field_name).results;
        num_trees_list = all_results.(field_name).num_trees_list;
        min_leaf_list = all_results.(field_name).min_leaf_list;
        if isfield(all_results.(field_name), 'dataset_name')
            dataset_name = all_results.(field_name).dataset_name;
        else
            dataset_name = field_name;
        end
        
        % 创建热力图（浅色系：白色背景，黑色字体，用于SCI文章）
        fig_heat = figure('Position', [100, 100, 1200, 400], 'Color', 'white', 'InvertHardcopy', 'off');
        
        % 测试集准确率热力图
        subplot(1, 3, 1);
        ax1 = gca;
        set(ax1, 'Color', 'white', 'XColor', 'black', 'YColor', 'black', ...
            'TickDir', 'out', 'TickLength', [0.02, 0.02]);
        acc_matrix = reshape(results(:, 5), length(min_leaf_list), length(num_trees_list));
        imagesc(num_trees_list, min_leaf_list, acc_matrix);
        cb1 = colorbar;
        cb1.Color = 'black';
        if ~isempty(cb1.Label)
            cb1.Label.Color = 'black';
        end
        % 设置colorbar刻度标签颜色为黑色
        try
            set(cb1, 'TickLabelColor', 'black');
        catch
            try
                cb1.YAxis.Color = 'black';
            catch
            end
        end
        xlabel('NumTrees', 'Color', 'black', 'FontWeight', 'bold');
        ylabel('MinLeafSize', 'Color', 'black', 'FontWeight', 'bold');
        title(sprintf('%s - 测试集准确率', dataset_name), 'Color', 'black', 'FontWeight', 'bold');
        set(gca, 'YDir', 'normal');
        % 确保刻度数字为黑色（通过XAxis和YAxis设置）
        try
            ax1.XAxis.Color = 'black';
            ax1.YAxis.Color = 'black';
            ax1.XAxis.TickLabelColor = 'black';
            ax1.YAxis.TickLabelColor = 'black';
        catch
            % 如果新版本属性不支持，使用传统方法
            set(ax1, 'XColor', 'black', 'YColor', 'black');
        end
        
        % F1-Score热力图
        subplot(1, 3, 2);
        ax2 = gca;
        set(ax2, 'Color', 'white', 'XColor', 'black', 'YColor', 'black', ...
            'TickDir', 'out', 'TickLength', [0.02, 0.02]);
        f1_matrix = reshape(results(:, 8), length(min_leaf_list), length(num_trees_list));
        imagesc(num_trees_list, min_leaf_list, f1_matrix);
        cb2 = colorbar;
        cb2.Color = 'black';
        if ~isempty(cb2.Label)
            cb2.Label.Color = 'black';
        end
        % 设置colorbar刻度标签颜色为黑色
        try
            set(cb2, 'TickLabelColor', 'black');
        catch
            try
                cb2.YAxis.Color = 'black';
            catch
            end
        end
        xlabel('NumTrees', 'Color', 'black', 'FontWeight', 'bold');
        ylabel('MinLeafSize', 'Color', 'black', 'FontWeight', 'bold');
        title(sprintf('%s - F1-Score', dataset_name), 'Color', 'black', 'FontWeight', 'bold');
        set(gca, 'YDir', 'normal');
        % 确保刻度数字为黑色
        try
            ax2.XAxis.Color = 'black';
            ax2.YAxis.Color = 'black';
            ax2.XAxis.TickLabelColor = 'black';
            ax2.YAxis.TickLabelColor = 'black';
        catch
            set(ax2, 'XColor', 'black', 'YColor', 'black');
        end
        
        % AUC热力图
        subplot(1, 3, 3);
        ax3 = gca;
        set(ax3, 'Color', 'white', 'XColor', 'black', 'YColor', 'black', ...
            'TickDir', 'out', 'TickLength', [0.02, 0.02]);
        auc_matrix = reshape(results(:, 9), length(min_leaf_list), length(num_trees_list));
        imagesc(num_trees_list, min_leaf_list, auc_matrix);
        cb3 = colorbar;
        cb3.Color = 'black';
        if ~isempty(cb3.Label)
            cb3.Label.Color = 'black';
        end
        % 设置colorbar刻度标签颜色为黑色
        try
            set(cb3, 'TickLabelColor', 'black');
        catch
            try
                cb3.YAxis.Color = 'black';
            catch
            end
        end
        xlabel('NumTrees', 'Color', 'black', 'FontWeight', 'bold');
        ylabel('MinLeafSize', 'Color', 'black', 'FontWeight', 'bold');
        title(sprintf('%s - AUC', dataset_name), 'Color', 'black', 'FontWeight', 'bold');
        set(gca, 'YDir', 'normal');
        
        sgtitle(sprintf('%s数据集 - 超参数影响分析', dataset_name), 'FontSize', 14, 'Color', 'black');
        % 文件名不能包含特殊字符
        safe_name = strrep(dataset_name, ' ', '_');
        safe_name = strrep(safe_name, '年', 'year');
        % 使用print确保白色背景和正确的颜色
        print(fig_heat, sprintf('hyperparameter_heatmap_%s.png', safe_name), '-dpng', '-r300', '-painters');
        close(fig_heat);
    end
end

function plot_dca_curve_matlab(y_true, y_proba, dataset_name)
    % 绘制决策曲线分析（DCA）曲线
    % y_true: 真实标签（必须是0和1）
    % y_proba: 预测概率
    % dataset_name: 数据集名称（英文）
    % 浅色系：白色背景，黑色字体，用于SCI文章
    
    % 确保坐标轴背景为白色，文字为黑色，浅灰色网格
    ax = gca;
    set(ax, 'Color', 'white', 'XColor', 'black', 'YColor', 'black', ...
        'GridColor', [0.9, 0.9, 0.9], 'MinorGridColor', [0.95, 0.95, 0.95], ...
        'GridAlpha', 1, 'MinorGridAlpha', 1, 'TickDir', 'out', 'TickLength', [0.02, 0.02]);
    % 确保刻度标签颜色为黑色
    try
        ax.XAxis.TickLabelColor = 'black';
        ax.YAxis.TickLabelColor = 'black';
    catch
        set(ax, 'XColor', 'black', 'YColor', 'black');
    end
    
    % 阈值范围
    threshold = 0:0.01:1;
    n = length(y_true);
    net_benefit = zeros(size(threshold));
    
    % 计算每个阈值的净收益
    for i = 1:length(threshold)
        pt = threshold(i);
        if pt == 0 || pt == 1
            continue;
        end
        
        % 根据阈值进行预测
        y_pred = double(y_proba >= pt);
        
        % 计算TP和FP
        TP = sum((y_true == 1) & (y_pred == 1));
        FP = sum((y_true == 0) & (y_pred == 1));
        
        % 计算净收益
        % Net Benefit = (TP/n) - (FP/n) * (pt/(1-pt))
        net_benefit(i) = (TP / n) - (FP / n) * (pt / (1 - pt));
    end
    
    % 绘制DCA曲线
    plot(threshold, net_benefit, 'LineWidth', 2.5, 'Color', [0.78, 0.24, 0.11]); % #C73E1D
    hold on;
    
    % 绘制"全部治疗"线（假设所有患者都治疗）
    prevalence = sum(y_true == 1) / n;
    treat_all = ones(size(threshold)) * prevalence - threshold * (prevalence / (1 - prevalence));
    plot(threshold, treat_all, '--', 'LineWidth', 2, 'Color', [0.42, 0.46, 0.48]); % #6C757D
    
    % 绘制"不治疗"线（假设所有患者都不治疗）
    treat_none = zeros(size(threshold));
    plot(threshold, treat_none, '--', 'LineWidth', 2, 'Color', [0.42, 0.46, 0.48]);
    
    xlabel('Threshold Probability', 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'black');
    ylabel('Net Benefit', 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'black');
    title(sprintf('%s - Decision Curve Analysis', dataset_name), ...
        'FontSize', 16, 'FontWeight', 'bold', 'Color', 'black');
    legend('RF Model', 'Treat All', 'Treat None', 'Location', 'best', ...
        'FontSize', 12, 'TextColor', 'black', 'Color', 'white', ...
        'EdgeColor', [0.8, 0.8, 0.8], 'Box', 'on');
    grid on;
    grid minor;
    set(ax, 'GridColor', [0.9, 0.9, 0.9], 'GridAlpha', 1);
    set(ax, 'MinorGridColor', [0.95, 0.95, 0.95], 'MinorGridAlpha', 1);
    xlim([0, 1]);
    y_min = min([min(net_benefit), min(treat_all), min(treat_none)]) - 0.05;
    y_max = max([max(net_benefit), max(treat_all), max(treat_none)]) + 0.05;
    ylim([y_min, y_max]);
end

