classdef Rls2 < Agent_technique2
    %{ Recursive Least Squares (RLS) Algorithm - State Estimation Version
    %  Based on Adaptive Filtering Algorithms and Practical Implementation, Chap.5, Diniz
    %  Refactored to estimate state vector x (consistent with Kalman filtering)
    %  Mathematical model: y = H_matrix * x + noise (estimate x, H_matrix known)
    %}
    properties (Access = public)
        P                   % state error covariance matrix (x_dim x x_dim)
        g                   % gain vector (x_dim x y_dim, similar to Kalman gain)
        alpha               % innovation vector (y_dim x 1)
        H_matrix            % known observation matrix (y_dim x x_dim)
        x_hat               % state estimate (x_dim x 1)
        lambda              % forgetting factor (0 < lambda <= 1)
        start_vals          % initial values structure
        iteracts            % number of iterations (time steps)
        y_hat               % predicted observation (y_dim x 1)
    end

    methods (Access = public)
        function obj = Rls2(varargin)
            % Constructor for RLS2 class
            % Inputs: 'x_dim', 'y_dim', 'H_matrix', 'lambda', 'start_vals'

            DEBUG(varargin)
            obj@Agent_technique2(varargin{:});

            p = inputParser;
            p.KeepUnmatched = true;

            % Default H_matrix (known observation matrix)
            default_H_matrix = eye(obj.y_dim, obj.x_dim);
            check_H_matrix = @(x) isnumeric(x) && all(size(x) == [obj.y_dim, obj.x_dim]);
            addParameter(p, 'H_matrix', default_H_matrix, check_H_matrix);

            % Default forgetting factor
            default_lambda = 0.9;
            check_lambda = @(x) isnumeric(x) && isscalar(x) && (x > 0) && (x <= 1);
            addParameter(p, 'lambda', default_lambda, check_lambda);

            % Default start_vals structure for state estimation
            default_start_vals = struct('delta', 0.1, 'initial_state', zeros(obj.x_dim, 1));
            check_start_vals = @(x) isstruct(x) && isfield(x, 'delta') && isfield(x, 'initial_state');
            addParameter(p, 'start_vals', default_start_vals, check_start_vals);

            try
                parse(p, varargin{:});

                % Initialize RLS parameters for state estimation
                obj.H_matrix = p.Results.H_matrix;
                obj.lambda = p.Results.lambda;
                obj.start_vals = p.Results.start_vals;

                % Validate start_vals structure %TODO: Create a more consistent validation (it firsts put the results then check, but need to first check and then put the results, but it is not a big problem since start_vals is only used to initialize the object)
                if ~isscalar(obj.start_vals.delta) || obj.start_vals.delta <= 0
                    error('Rls2: start_vals.delta must be a positive scalar');
                end

                if length(obj.start_vals.initial_state) ~= obj.x_dim
                    error('Rls2: start_vals.initial_state must have length equal to x_dim');
                end

                % Initialize state estimation variables
                obj.x_hat = reshape(obj.start_vals.initial_state, obj.x_dim, 1);
                obj.P = obj.start_vals.delta * eye(obj.x_dim);  % State covariance
                obj.g = zeros(obj.x_dim, obj.y_dim);           % State gain matrix
                obj.alpha = zeros(obj.y_dim, 1);               % Innovation vector
                obj.y_hat = zeros(obj.y_dim, 1);               % Predicted observation
                obj.iteracts = 0;

                obj.xp_hat = obj.x_hat;
                obj.xa_hat = obj.x_hat;

            catch exception
                error('An error occurred in Rls2 constructor: %s', exception.message);
            end

            DEBUG(obj.H_matrix)
            DEBUG(obj.x_hat)
            DEBUG(obj.lambda)
            DEBUG(obj.start_vals)
            DEBUG(obj.P)
        end

        function [varargout] = apply(obj, varargin)
            % Apply RLS state estimation technique
            % Input: 'measurement', measurement_value
            % Output: x_hat (state estimate), y_hat (predicted observation)
            % Mathematical model: y = H_matrix * x + noise (estimate x)

            DEBUG(varargin)

            try
                % Parse input arguments - only measurement needed for state estimation
                ind = find(strcmp(varargin, 'measurement'));
                if isempty(ind)
                    error('Rls2: Measurement not provided.');
                else
                    y = varargin{ind+1};
                    if ~isnumeric(y) || length(y) ~= obj.y_dim
                        error('Rls2: Measurement must be numeric with length equal to y_dim');
                    end
                    y = reshape(y, obj.y_dim, 1); % Ensure column vector
                end

                obj.iteracts = obj.iteracts + 1;

                % RLS State Estimation Algorithm
                % Prior prediction using current state estimate
                y_hat_prior = obj.H_matrix * obj.x_hat;

                % Innovation (prediction error)
                obj.alpha = y - y_hat_prior;

                % Gain matrix calculation (for state estimation)
                obj.g = obj.P * obj.H_matrix' / (obj.lambda + obj.H_matrix * obj.P * obj.H_matrix');

                % State covariance matrix update
                obj.P = (1/obj.lambda) * (obj.P - obj.g * obj.H_matrix * obj.P);

                % Divergence protection
                if norm(obj.P, 'fro') >= 1e15
                    obj.reset_P();
                    obj.update_lambda(0.001);
                end

                % State update (estimate x)
                obj.x_hat = obj.x_hat + obj.g * obj.alpha;

                obj.xp_hat = obj.x_hat;
                obj.xa_hat = obj.xp_hat;

                % Posterior prediction
                obj.y_hat = obj.H_matrix * obj.x_hat;

                % Return outputs based on number of requested outputs
                switch nargout
                    case 0
                        fprintf('Rls2: No output requested in RLS.\n');
                    case 1
                        varargout{1} = obj.y_hat;
                    case 2
                        varargout{1} = obj.y_hat;
                        varargout{2} = obj.x_hat;
                    case 3
                        varargout{1} = obj.y_hat;
                        varargout{2} = obj.x_hat;
                        varargout{3} = obj.alpha;
                    case 4
                        varargout{1} = obj.y_hat;
                        varargout{2} = obj.x_hat;
                        varargout{3} = obj.alpha;
                        varargout{4} = obj.g;
                    case 5
                        varargout{1} = obj.y_hat;
                        varargout{2} = obj.x_hat;
                        varargout{3} = obj.alpha;
                        varargout{4} = obj.g;
                        varargout{5} = obj.P;
                    otherwise
                        error('RLS2::apply : Too many outputs requested.');
                end

            catch exception
                error('An error occurred in RLS apply: %s', exception.message);
            end
        end

        function obj = reset(obj)
            % Reset the RLS filter to initial conditions for state estimation
            try
                obj.x_hat = reshape(obj.start_vals.initial_state, obj.x_dim, 1);
                obj.P = obj.start_vals.delta * eye(obj.x_dim);
                obj.g = zeros(obj.x_dim, obj.y_dim);
                obj.alpha = zeros(obj.y_dim, 1);
                obj.y_hat = zeros(obj.y_dim, 1);
                obj.iteracts = 0;
            catch exception
                error('Rls2: An error occurred in RLS reset: %s', exception.message);
            end
        end

        function obj = update_params(obj, varargin)
            % Update RLS parameters for state estimation
            % Supported parameters: 'H_matrix', 'lambda', 'covariance_estimate', 'state_estimate'

            try
                ind = find(strcmp(varargin, 'H_matrix'));
                if ~isempty(ind)
                    H_new = varargin{ind+1};
                    if all(size(H_new) == [obj.y_dim, obj.x_dim])
                        obj.H_matrix = H_new;
                    else
                        error('Rls2: H_matrix dimensions must be [y_dim, x_dim]');
                    end
                end

                ind = find(strcmp(varargin, 'lambda'));
                if ~isempty(ind)
                    lambda_new = varargin{ind+1};
                    if isscalar(lambda_new) && lambda_new > 0 && lambda_new <= 1
                        obj.lambda = lambda_new;
                    else
                        error('Rls2: lambda must be a scalar in (0, 1]');
                    end
                end

                ind = find(strcmp(varargin, 'covariance_estimate'));
                if ~isempty(ind)
                    P_new = varargin{ind+1};
                    if all(size(P_new) == [obj.x_dim, obj.x_dim])
                        obj.P = P_new;
                    else
                        error('Rls2: covariance_estimate dimensions must be [x_dim, x_dim]');
                    end
                end

                ind = find(strcmp(varargin, 'state_estimate'));
                if ~isempty(ind)
                    x_new = varargin{ind+1};
                    if length(x_new) == obj.x_dim
                        obj.x_hat = reshape(x_new, obj.x_dim, 1);
                    else
                        error('Rls2: state_estimate length must equal x_dim');
                    end
                end

            catch exception
                error('An error occurred in RLS update_params: %s', exception.message);
            end
        end

        function params = get_params(obj, varargin)
            % Get RLS parameters for state estimation
            % Supported parameters: 'H_matrix', 'lambda', 'covariance_estimate', 'prediction', 'state_estimate'

            params = struct();
            try
                ind = find(strcmp(varargin, 'H_matrix'));
                if ~isempty(ind)
                    params.H_matrix = obj.H_matrix;
                end

                ind = find(strcmp(varargin, 'lambda'));
                if ~isempty(ind)
                    params.lambda = obj.lambda;
                end

                ind = find(strcmp(varargin, 'covariance_estimate'));
                if ~isempty(ind)
                    params.covariance_estimate = obj.P;
                end

                ind = find(strcmp(varargin, 'prediction'));
                if ~isempty(ind)
                    params.prediction = obj.y_hat;
                end

                ind = find(strcmp(varargin, 'state_estimate'));
                if ~isempty(ind)
                    params.state_estimate = obj.x_hat;
                end

                % If no specific parameter requested, return all
                if isempty(varargin)
                    params.H_matrix = obj.H_matrix;
                    params.lambda = obj.lambda;
                    params.covariance_estimate = obj.P;
                    params.prediction = obj.y_hat;
                    params.state_estimate = obj.x_hat;
                    params.iterations = obj.iteracts;
                end

            catch exception
                error('An error occurred in RLS get_params: %s', exception.message);
            end
        end

        % Additional utility methods for state estimation
        function H_matrix = get_H_matrix(obj)
            % Get the observation matrix (known parameter)
            H_matrix = obj.H_matrix;
        end

        function obj = update_H_matrix(obj, H_new)
            % Update the observation matrix
            if all(size(H_new) == [obj.y_dim, obj.x_dim])
                obj.H_matrix = H_new;
            else
                error('Rls2: H matrix dimensions must be [y_dim, x_dim]');
            end
        end

        function x_hat = get_x_hat(obj)
            % Get current state estimate
            x_hat = obj.x_hat;
        end

        function obj = update_x_hat(obj, x_new)
            % Update state estimate
            if length(x_new) == obj.x_dim
                obj.x_hat = reshape(x_new, obj.x_dim, 1);
            else
                error('Rls2: State vector length must equal x_dim');
            end
        end

        function y_hat = get_y_hat(obj, state_vector)
            % Get predicted observation for given state vector or current estimate
            if nargin < 2
                y_hat = obj.y_hat;
            else
                if length(state_vector) ~= obj.x_dim
                    error('State vector length must equal x_dim');
                end
                state_vector = reshape(state_vector, obj.x_dim, 1);
                y_hat = obj.H_matrix * state_vector;
            end
        end

        % Backward compatibility methods (deprecated - use state estimation methods)
        function H = get_H(obj)
            % DEPRECATED: Use get_H_matrix() instead
            % Returns H_matrix for backward compatibility
            warning('get_H() is deprecated. Use get_H_matrix() for observation matrix or get_x_hat() for state estimate.');
            H = obj.H_matrix;
        end

        function obj = update_H(obj, H_new)
            % DEPRECATED: Use update_H_matrix() instead
            warning('update_H() is deprecated. Use update_H_matrix() for observation matrix or update_x_hat() for state estimate.');
            obj = obj.update_H_matrix(H_new);
        end

        function obj = reset_P(obj)
            % Reset covariance matrix to initial value
            obj.P = obj.start_vals.delta * eye(obj.x_dim);
        end

        function obj = update_lambda(obj, delta_lambda)
            % Update lambda with bounds checking
            new_lambda = obj.lambda + delta_lambda;
            if new_lambda > 0 && new_lambda <= 1
                obj.lambda = new_lambda;
            else
                warning('Lambda update would violate bounds (0, 1], keeping current value');
            end
        end

        function iterations = get_iterations(obj)
            % Get number of iterations
            iterations = obj.iteracts;
        end

    end % methods
end % classdef