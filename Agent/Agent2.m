classdef Agent2 < handle
    properties
        y_hat               % Variável de observação que se estima a partir de H
        % H_hat               % Variável de pesos que estima (for compatibility)
        % x_dim               % Dimensão do vetor estado
        % y_dim               % Dimensão do vetor observação
        agent_technique         % Técnica de update adaptativo (Agent_technique2)
        % Kalman-specific properties
        xp_hat                  % Posterior state estimate
        xa_hat                  % Prior state estimate
        last_measurement        % Last measurement received
        % Fusion-specific properties
        neighbors               % Neighbors information
        fusion_technique        % Fusion technique
        neighbors_weights       % Weights for neighbor information
        fusion_results struct   % Results from self and social learning
    end
    properties (SetAccess = private)
        ID double; % Unique identifier for the object
    end

    methods
        function obj = Agent2(varargin)
            
            persistent uniqueID; % Persistent variable to keep track of unique IDs across all instances
            
            p = inputParser;
            p.KeepUnmatched = true;

            % default_x_dim = 3;
            % check_x_dim = @(x) isnumeric(x) && isscalar(x) && (x > 0) && (mod(x,1)==0);
            % addOptional(p, 'x_dim', default_x_dim, check_x_dim);

            % default_y_dim = 1;
            % check_y_dim = @(x) isnumeric(x) && isscalar(x) && (x > 0) && (mod(x,1)==0);
            % addOptional(p, 'y_dim', default_y_dim, check_y_dim);

            default_agent_tech = KF_diff();
            check_agent_tech = @(x) isa(x, "Agent_technique2");
            addOptional(p, 'agent_tech', default_agent_tech, check_agent_tech);

            default_fusion_tech = General_Adapt_and_Fuse();
            check_fusion_tech = @(x) isa(x, "Fusion_technique2");
            addOptional(p, 'fusion_tech', default_fusion_tech, check_fusion_tech);


            try
                parse(p, varargin{:});
                % obj.x_dim = p.Results.x_dim;
                % obj.y_dim = p.Results.y_dim;
                obj.agent_technique = p.Results.agent_tech;
                obj.fusion_technique = p.Results.fusion_tech;

                % Initialize Kalman-specific properties
                obj.y_hat = zeros(obj.agent_technique.y_dim, 1);
                % obj.H_hat = zeros(obj.agent_technique.y_dim, obj.agent_technique.x_dim);  % For compatibility with existing fusion techniques
                obj.xp_hat = zeros(obj.agent_technique.x_dim, 1);
                obj.xa_hat = zeros(obj.agent_technique.x_dim, 1);
                obj.last_measurement = [];

                obj.neighbors = [];
                obj.neighbors_weights = [];
                obj.fusion_results = struct();
                obj.fusion_results.state_estimate = [];
                obj.fusion_results.covariance_estimate = [];

                % % Ensure agent_technique dimensions match
                % if obj.agent_technique.x_dim ~= obj.x_dim || obj.agent_technique.y_dim ~= obj.y_dim
                %     error('Agent2: agent_technique dimensions must match agent dimensions');
                % end
                
                if isempty(uniqueID)
                    uniqueID = 0; % Initialize the unique ID counter
                end
                uniqueID = uniqueID + 1; % Increment the unique ID for each new instance
                obj.ID = uniqueID; % Assign the incremented value to the object's ID property

            catch exception
                error('An error occurred %s\n', exception.message);
            end
        end

        function obj = reset(obj)
            % Reset agent state
            obj.y_hat = zeros(obj.agent_technique.y_dim, 1);
            % obj.H_hat = zeros(obj.agent_technique.y_dim, obj.agent_technique.x_dim);
            obj.xp_hat = zeros(obj.agent_technique.x_dim, 1);
            obj.xa_hat = zeros(obj.agent_technique.x_dim, 1);
            obj.last_measurement = [];
            obj.fusion_results.state_estimate = [];
            obj.fusion_results.covariance_estimate = [];
            % Reset the agent technique
            obj.agent_technique.reset();
        end

        function obj = self_learning_step(obj, varargin)
            % Self-learning step for Kalman filtering
            % measurement: current observation
            % varargin: additional parameters for Kalman filtering
            
            % try
            %     ind = find(strcmp(varargin, 'measurement'));
            %     if isempty(ind)
            %         error('Measurement not provided.')
            %     else
            %         measurement = varargin{ind+1};
            %     end
            % catch exception
            %     error('An error occurred: %s', exception.message);
                
            % end

            % obj.last_measurement = measurement;
            
            % Apply Kalman filtering technique
            DEBUG(varargin)
            [~,~,~,~,~,~,~,obj.y_hat] = obj.agent_technique.apply(varargin{:});
            
            % FIXME: Wrong pattern breaking the OOP principles
            % % Update state estimates from Kalman filter
            % switch class(obj.agent_technique)
            %     case 'Kalman2'
            %         obj.xp_hat = obj.agent_technique.xp_hat;
            %         obj.xa_hat = obj.agent_technique.xa_hat;

            %     otherwise
            %         DEBUG(obj.agent_technique)
            %         error('Agent2: Only Kalman2 is supported for now.')
            % end
            % Update agent state estimates
            obj.xp_hat = obj.agent_technique.xp_hat;
            obj.xa_hat = obj.agent_technique.xa_hat;

            % if isa(obj.agent_technique, 'Kalman2')
                
                
            %     % Update H_hat for compatibility with fusion techniques
            %     % Extract observation matrix from Kalman filter if available
            %     % if isprop(obj.agent_technique, 'H') || isfield(obj.agent_technique, 'H')
            %     %     obj.H_hat = obj.agent_technique.H;
            %     % end
            % end
        end
        
        function obj = social_learning_step(obj)

            % p=inputParser;
            % p.KeepUnmatched = true;

            obj.fusion_technique.social_learning_step('self_agent', obj, ...
                                                      'dim', obj.agent_technique.x_dim);      

            DEBUG(obj.getID())
            DEBUG(obj.fusion_results.state_estimate)
            DEBUG(obj.fusion_results.covariance_estimate)
            % Social learning step - update based on neighbor information
            % For Kalman filters, this could update the observation matrix H
            % obj.H_hat = new_H_hat;
            
            % Update the agent technique if it supports H matrix updates
            % if hasMethod(obj.agent_technique, 'update_H')
            %     obj.agent_technique.update_H(obj.H_hat);
            % elseif isprop(obj.agent_technique, 'H') || isfield(obj.agent_technique, 'H')
            %     obj.agent_technique.H = obj.H_hat;
            % end
        end

        function update_agent_estimates(obj)
            obj.fusion_technique.update_agent_technique_params(obj);
        end
        
        function obj = add_agent(obj, agent, varargin)
            obj.neighbors = [obj.neighbors; agent];

            obj.fusion_technique.reevaluate_weights(varargin{:});
            
        end

        function obj = remove_agent(obj, agent_idx)
            % Remove agent from neighbors list
            find_idx = find(obj.neighbors.ID == agent_idx);
            if ~isempty(find_idx)
                obj.neighbors(find_idx) = [];
                obj.fusion_technique.reevaluate_weights();
            else
                error('Fusion_technique2: Error in remove_agent - Agent not found.');
            end
            
        end

        function y_hat = get_y_hat(obj)
            y_hat = obj.y_hat;
        end

        % function H_hat = get_H_hat(obj)
        %     H_hat = obj.H_hat;
        % end

        function xp_hat = get_posterior_state(obj)
            % Get posterior state estimate
            xp_hat = obj.xp_hat;
        end

        function xa_hat = get_prior_state(obj)
            % Get prior state estimate
            xa_hat = obj.xa_hat;
        end

        function measurement = get_last_measurement(obj)
            % Get last measurement
            measurement = obj.last_measurement;
        end

        % Additional Kalman-specific methods
        function P = get_posterior_covariance(obj)
            % Get posterior error covariance matrix
            try %isprop(obj.fusion_results, 'covariance_estimate') && 
                
                if ~isempty(obj.fusion_results.covariance_estimate) % TODO: do a more general algorithm, in order to get a matrix or in order to get a covariance estimate for all kind of filter?
                    P = obj.fusion_results.covariance_estimate;
                end
            catch exception
                error('Agent2: Error in get_posterior_covariance - %s', exception.message);
            end
            % Este tipo de implementação vai contra os princípios básicos de OOP
            % if isa(obj.agent_technique, 'Kalman2') && isprop(obj.agent_technique, 'Pp')
            %     P = obj.agent_technique.Pp;
            % else
            %     P = [];
            % end

        end

        function P = get_prior_covariance(obj)
            % Get prior error covariance matrix
            if isa(obj.agent_technique, 'Kalman2') && isprop(obj.agent_technique, 'Pa')
                P = obj.agent_technique.Pa;
            else
                P = [];
            end
        end

        function K = get_kalman_gain(obj)
            % Get Kalman gain matrix
            if isa(obj.agent_technique, 'Kalman2') && isprop(obj.agent_technique, 'K')
                K = obj.agent_technique.K;
            else
                K = [];
            end
        end

        function ID = getID(obj)
            ID = obj.ID;
        end
    end
end
