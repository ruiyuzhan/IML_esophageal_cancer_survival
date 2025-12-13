% =========================================================================
% File Description: Process Random Forest (TreeBagger) prediction results,
%                   converting cell arrays or string-format predictions to
%                   numeric label arrays.
% Copyright (c) [YEAR] [AUTHOR]. All rights reserved.
% =========================================================================

function y_processed = RF_process(y_predict)
% Process Random Forest prediction results
% Input: y_predict - TreeBagger prediction results (may be cell array or numeric array)
% Output: y_processed - Processed numeric label array

% If input is cell array (TreeBagger classification results are usually cell arrays)
if iscell(y_predict)
    y_processed = zeros(length(y_predict), 1);
    for i = 1:length(y_predict)
        if ischar(y_predict{i}) || isstring(y_predict{i})
            % If string, convert to numeric
            y_processed(i) = str2double(y_predict{i});
        elseif isnumeric(y_predict{i})
            % If already numeric, use directly
            y_processed(i) = y_predict{i};
        else
            % Other cases, try to convert
            y_processed(i) = str2double(char(y_predict{i}));
        end
        
        % If conversion fails, set to NaN (can be handled later)
        if isnan(y_processed(i))
            y_processed(i) = 0;  % Default value
        end
    end
else
    % If input is already numeric array, return directly
    y_processed = double(y_predict(:));  % Ensure column vector
end

% Ensure output is integer (classification labels are usually integers)
y_processed = round(y_processed);

end

