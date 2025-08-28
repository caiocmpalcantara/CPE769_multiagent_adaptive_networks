classdef Kalman2 < Agent_technique2
    % Class: Kalman2 - Represents the General Kalman filtering technique (abstract)
    % This Kalman assumption is that the System Model is always linear, so the system_model can define every
    % kind of possible models and contain the every system parameters.
    % In that sense, the Kalman filtering technique will only deal with Sensor Model in its own way (concrete class).
    properties (Access = public)
        system_model            % Kalman System Model
        % y_dim                   % Dimension of the observation -- is already in Agent_technique
        xp_hat                  % State prior estimate
        xa_hat                  % State posterior estimate
        a                       % Innovation
        last_T_sample_time      % Time between samples
        last_update_timestamp   % Timestamp of the last sample
        % Statistics
        Pp                      % Posterior Error covariance matrix
        Pa                      % Prior Error covariance matrix
        S                       % Innovation matrix
        K                       % Kalman gain matrix
        iteracts                % Number of iteractions
    end

    methods (Abstract)
        y = update_obs(obj, measurement);
        y_hat = update_y_hat(obj, varargin);
        Py = update_Py(obj, varargin);
        Pxy = update_Pxy(obj, varargin);
        R = update_R_matrix(obj, varargin);
    
        % Px = update_Px(obj, time);
        % Q = update_Q_matrix(obj, varargin);
        % [varargout] = apply(obj, varargin)    From Agent_technique: apply filtering technique for each obs sample
        
    end

    methods
        function obj = Kalman2(varargin)
            % obj = Kalman2(pos0, pos1, t0, t1, range_sd, bearing_sd)
            
            DEBUG(varargin)
            obj@Agent_technique2(varargin{:}); % TODO: Create a arugment fusion that captures the system model x dimension ??? What does come first?

            % DEBUG(obj.y_dim)

            p = inputParser;
            p.KeepUnmatched = true;

            default_system_model = Linear_State('dim', obj.x_dim);
            check_system_model = @(x) isa(x, "System_Model");
            addOptional(p, 'system_model', default_system_model, check_system_model);

            default_last_update_timestamp = datetime('now');
            check_last_update_timestamp = @(x) isa(x, "datetime");
            addOptional(p, 'last_update_timestamp', default_last_update_timestamp, check_last_update_timestamp);

            default_last_T_sample_time = 1;
            check_last_T_sample_time = @(x) isnumeric(x) && isscalar(x) && (x>0);
            addOptional(p, 'last_T_sample_time', default_last_T_sample_time, check_last_T_sample_time);

            default_Pa_init = {'delta', 0.1};
            check_Pa_init = @(x) isa(x, 'cell');
            addOptional(p, 'Pa_init', default_Pa_init, check_Pa_init);

            default_xa_init = {};   % it must be the default option in method
            check_xa_init = @(x) isa(x, 'cell');
            addOptional(p, 'xa_init', default_xa_init, check_xa_init);

            % default_y_dim = 1; %FIXME: Already in Agent_technique2
            % check_y_dim = @(x) isnumeric(x) && isscalar(x) && (x>0) && (mod(x,1)==0);
            % addOptional(p, 'y_dim', default_y_dim, check_y_dim);

            try
                parse(p, varargin{:});
                obj.system_model = p.Results.system_model;
                obj.Pa = obj.system_model.Pa_init(p.Results.Pa_init{:});
                obj.Pp = obj.Pa;
                obj.xa_hat = obj.system_model.xa_init(p.Results.xa_init{:});
                obj.xp_hat = obj.xa_hat;
                % obj.y_dim = p.Results.y_dim;
                obj.a = zeros(obj.y_dim, 1);
                obj.S = zeros(obj.y_dim);
                obj.K = zeros(obj.system_model.dim, obj.y_dim); 
                obj.iteracts = 0;
            catch exception
                error("An error occurred: %s", exception.message);                
            end

            DEBUG(obj.system_model)
            DEBUG(obj.Pa)
            DEBUG(obj.Pp)
            DEBUG(obj.xa_hat)
            DEBUG(obj.xp_hat)
            DEBUG(obj.a)
            DEBUG(obj.S)
            DEBUG(obj.K)
            DEBUG(obj.iteracts)
            % obj.VAR_R = range_sd.^2;
            % obj.VAR_phi = bearing_sd.^2;
            
            % obj.S = zeros(2,2);
            % obj.K = zeros(4,2);
            % obj.a = [0;0];
            
        end

        % Method: prior_estimate - Gives the prior estimate
        % -------------------------------------------------
        function xa_hat = prior_estimate(obj, varargin)
            % @brief Calculate the prior state estimate in double vector form
            try
                A = obj.system_model.update_A(varargin);
                xa_hat = A * obj.xp_hat;
                obj.xa_hat = xa_hat;
            catch exception
                error("Unexpected object type: %s", exception.message);
            end
        end

        % Method: update_sample_time - obtain the time between samples
        % ------------------------------------------------------------
        function last_T_sample_time = update_sample_time(obj, timestamp)
            % @brief Calculate the time between the last sample and the new aquired and
            % update the last update timestamp.

            try
                if ~isa(timestamp, "datetime")
                    error('The input timestamp must be a datetime object.')
                elseif obj.last_update_timestamp < timestamp
                    last_T_sample_time = seconds(timestamp - obj.last_update_timestamp);
                    obj.last_T_sample_time = last_T_sample_time;
                    obj.last_update_timestamp = timestamp;
                else
                    last_T_sample_time = obj.last_T_sample_time;
                end
            catch exception
                error("An error occurred: %s", exception.message)
            end
        end

        % Method: update_A - update the matrix A
        % ----------------------------------------
        function A = update_A(obj, varargin)
            % @brief Update the matrix A with variables needed to the respective system model
            % If timestamp is provided, it updates the sample time and gives it to update A.
            % For efficiency, it updates the object last_update_timestamp and last_T_sample_time.
            DEBUG(varargin)
            try
                ind = find(strcmp(varargin, 'timestamp'));
                if ~isempty(ind)
                    T = obj.update_sample_time(varargin{ind+1}); % TODO: neeeds to enhance the processing (it is account 2x times in A and Q)
                    %append
                    varargin{end+1} = 'sample_time';
                    varargin{end+1} = T;

                    obj.last_update_timestamp = varargin{ind+1};
                    obj.last_T_sample_time = T;

                else
                    obj.last_update_timestamp = obj.last_update_timestamp + seconds(obj.last_T_sample_time);
                
                end
    
                A = obj.system_model.update_A(varargin{:});
            catch exception
                error('An error occurred: %s', exception.message);
            end
            % DEBUG(varargin)
        end
        
        % Method: update_Q - update the matrix Q
        % ----------------------------------------
        function Q = update_Q_matrix(obj, varargin)
            % @brief Update the matrix Q with variables needed to the respective system model
            % If timestamp is provided, it update the sample and gives it to the update Q.
            DEBUG(varargin)
            try
                % The update_A already added the sample_time to the varargin
                % ind = find(strcmp(varargin, 'timestamp'));
                % if ~isempty(ind)
                %     T = obj.update_sample_time(varargin{ind+1});
                %     varargin{end+1} = 'sample_time';
                %     varargin{end+1} = T;
                % end

                Q = obj.system_model.update_Q(varargin{:});
            catch exception
                error('An error occurred: %s', exception.message);
            end
            DEBUG(varargin)
        end
        
        % Method: update_Px - Update the matrix Px
        % ---------------------------------------------
        function Px = update_Px(obj, varargin)
            % @brief Update the matrix Px in the prior step.
            DEBUG(varargin)
            try
                % A = obj.system_model.update_A(varargin{:}); % TODO: Optimize the processing about A update globally at Kalman
                A = obj.system_model.get_A_matrix();
                Px = A * obj.Pp * (A)';    
            catch exception
                error('An error occurred: %s', exception.message)
            end
            
        end

        function obj = update_params(obj, varargin)
            % @brief Update the Kalman filtering parameters based on new information.
           
            try
                ind = find(strcmp(varargin, 'state_estimate'));
                if ~isempty(ind)
                    obj.xp_hat = varargin{ind+1};
                end

                ind = find(strcmp(varargin, 'covariance_estimate'));
                if ~isempty(ind)
                    obj.Pp = varargin{ind+1};
                end

            catch exception
                error('An error occurred: %s', exception.message)
            end
        end

        function params = get_params(obj, varargin)
            % @brief Get the Kalman filtering parameters.
            params = struct();
            try
                ind = find(strcmp(varargin, 'state_estimate'));
                if ~isempty(ind)
                    params.state_estimate = obj.xp_hat;
                end

                ind = find(strcmp(varargin, 'covariance_estimate'));
                if ~isempty(ind)
                    params.covariance_estimate = obj.Pp;
                end

            catch exception
                error('An error occurred: %s', exception.message)
            end
        end

        % Method: apply - From Agent_technique, apply the technique, updating the Kalman filtering statistics 
        % ---------------------------------------------------------
        function [varargout] = apply(obj, varargin)
            % @brief Update the Kalman filtering statistics based on new observed sample,
            % in order to iterate the Kalman filtering recursively following the Prior (System Model)
            % and Posterior (Sensor Model) steps.

            DEBUG(varargin)
            
            try
                ind = find(strcmp(varargin, 'measurement'));
                if isempty(ind)
                    error('Measurement not provided.')
                else
                    measurement = varargin{ind+1};
                end
            catch exception
                error('An error occurred: %s', exception.message)
            end

            % Prior
            A = obj.update_A(varargin{:});  %get last_update_timestamp and last_time_sample
            obj.xa_hat = A * obj.xp_hat;
            obj.Pa = obj.update_Px(varargin{:}) + obj.update_Q_matrix(varargin{:});

            % Update observation
            y = obj.update_obs(measurement);

            % Posterior
            S = obj.update_Py(varargin{:}) + obj.update_R_matrix(varargin{:});
            K = obj.update_Pxy(varargin{:}) / S;
            obj.Pp = obj.Pa - K * S * K';
            y_hat = obj.update_y_hat(varargin{:});
            a = y - y_hat;
            obj.xp_hat = obj.xa_hat + K * a;

            
            % obj.last_update_timestamp = varargin;

            % get statistics
            obj.S = S;
            obj.K = K;
            obj.a = a;

            switch nargout
                case 0
                    fprintf('No output requested in Kalman.\n');
                case 1  % xp_hat
                    varargout{1} = obj.xp_hat;
                case 2 % xp_hat and xa_hat
                    varargout{1} = obj.xp_hat;
                    varargout{2} = obj.xa_hat;
                case 3
                    varargout{1} = obj.xp_hat;
                    varargout{2} = obj.xa_hat;
                    varargout{3} = obj.Pp;
                case 4
                    varargout{1} = obj.xp_hat;
                    varargout{2} = obj.xa_hat;
                    varargout{3} = obj.Pp;
                    varargout{4} = obj.Pa;
                case 5
                    varargout{1} = obj.xp_hat;
                    varargout{2} = obj.xa_hat;
                    varargout{3} = obj.Pp;
                    varargout{4} = obj.Pa;
                    varargout{5} = obj.a;
                case 6
                    varargout{1} = obj.xp_hat;
                    varargout{2} = obj.xa_hat;
                    varargout{3} = obj.Pp;
                    varargout{4} = obj.Pa;
                    varargout{5} = obj.a;
                    varargout{6} = obj.S;
                case 7
                    varargout{1} = obj.xp_hat;
                    varargout{2} = obj.xa_hat;
                    varargout{3} = obj.Pp;
                    varargout{4} = obj.Pa;
                    varargout{5} = obj.a;
                    varargout{6} = obj.S;
                    varargout{7} = obj.K;
                case 8
                    varargout{1} = obj.xp_hat;
                    varargout{2} = obj.xa_hat;
                    varargout{3} = obj.Pp;
                    varargout{4} = obj.Pa;
                    varargout{5} = obj.a;
                    varargout{6} = obj.S;
                    varargout{7} = obj.K;
                    varargout{8} = y_hat;
                otherwise
                    error('Kalman::apply : Too many outputs.')
            end

        end

        function obj = update_agent_state_estimates(obj, agent)
            agent.xp_hat = obj.xp_hat;
            agent.xa_hat = obj.xa_hat;
        end

        function xa = get_prior_state(obj)
            xa = obj.xa_hat;
        end

        function xp = get_posterior_state(obj)
            xp = obj.xp_hat;
        end

        function set_priors(obj, varargin)
            p = inputParser;

            default_sig_R = 15;
            valid_sig_R = [1 1e2];
            check_sig_R = @(x) isnumeric(x) && all(and(x >= min(valid_sig_R), x <= max(valid_sig_R)));

            default_sig_phi = 2;
            valid_sig_phi = [.1 2e1];
            check_sig_phi = @(x) isnumeric(x) && all(and(x >= min(valid_sig_phi), x <= max(valid_sig_phi)));

            addOptional(p, 'sig_R', default_sig_R, check_sig_R);
            addOptional(p, 'sig_phi', default_sig_phi, check_sig_phi);

            parse(p, varargin{:});

            obj.VAR_R = (p.Results.sig_R).^2;
            obj.VAR_phi = (deg2rad(p.Results.sig_phi)).^2;
        end

        function obj = reset(obj)
            obj.K = zeros(obj.x_dim, obj.y_dim);
            obj.S = zeros(obj.y_dim);
            obj.iteracts = 0;
            obj.a = zeros(obj.x_dim, 1);
            obj.last_update_timestamp = [];
            obj.last_T_sample_time = [];
            obj.Pp = obj.system_model.Pa_init('delta', obj.system_model.initial_params.delta);
            obj.Pa = obj.Pp;
            obj.xa_hat = obj.system_model.xa_init('initial_state', obj.system_model.initial_params.initial_state);
            obj.xp_hat = obj.xa_hat;   
        end
        
    end

    methods (Static)
        function assign(kf1, kf2)
            try
                kf1.last_update_timestamp = kf2.last_update_timestamp;
                kf1.Pp = kf2.Pp;
                kf1.Pa = kf2.Pa;
                if(isa(kf1, 'PDNKF') && isa(kf2, 'PDNKF'))
                    kf1.Q = kf2.Q;
                end
                kf1.xp_hat = kf2.xp_hat;
                kf1.xa_hat = kf2.xa_hat;
                kf1.last_T_sample_time = kf2.last_T_sample_time;
            catch exception
                disp(exception);
            end 
        end
    end
end