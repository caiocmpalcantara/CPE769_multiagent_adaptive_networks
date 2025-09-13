classdef Fusion_technique2 < handle
    properties
        neighbors
        neighbors_weights
        % x_hat
    end
    methods (Abstract)
        %{ apply:
        %  Recebe a referência do vector de agentes e altera suas
        % estimativas com base no critério social.
        % Uses varargin for flexible parameter passing compatible with Agent_technique2
        %}
        apply(obj, varargin)
        % get_neighbours_estimates(obj)   %TODO: verify if will need varargin
    end

    methods
        function obj = Fusion_technique2(varargin)
            p = inputParser;
            p.KeepUnmatched = true;
            
            addParameter(p, 'neighbors', [], @(x) isa(x, 'Agent2'));
            addParameter(p, 'neighbors_weights', [], @(x) isnumeric(x));

            try
                parse(p, varargin{:});
                neighbors = p.Results.neighbors;
                neighbors_weights = p.Results.neighbors_weights;

                if ~isempty(neighbors) && ~isempty(neighbors_weights)
                    if length(neighbors) ~= length(neighbors_weights)
                        error('Fusion_technique2: Error in constructor - Number of neighbors must match number of weights.');
                    elseif sum(neighbors_weights) > 1+1e-5 && sum(neighbors_weights) < 1-1e-5
                        error('Fusion_technique2: Error in constructor - Weights must sum to 1.');
                    end
                end

                obj.neighbors = neighbors;
                obj.neighbors_weights = neighbors_weights;
                % obj.x_hat = [];
            catch exception
                error('Fusion_technique2: Error in constructor - %s', exception.message);
            end

            obj.neighbors = neighbors;
        end

        function obj = reevaluate_weights(obj, varargin)
            % Reevaluate weights based on current state and performance
            % Placeholder for future implementation
            % For now, keep existing weights
            % TODO: Implement reevaluation logic
        end

        function social_learning_step(obj, varargin)
            % Apply fusion technique and update agent parameters
            DEBUG(varargin)
            p = inputParser;
            p.KeepUnmatched = true;
            
            addParameter(p, 'self_agent', [], @(x) isa(x, 'Agent2'));
            addParameter(p, 'dim', [], @(x) isnumeric(x) && isscalar(x) && (x>0) && (mod(x,1)==0));

            try
                parse(p, varargin{:});
                agent = p.Results.self_agent;
                if strcmp('self_agent', p.UsingDefaults)
                    DEBUG(varargin)
                    error('Fusion_technique2: Error in social_learning_step - Self agent not provided.')
                % elseif strcmp('dim', p.UsingDefaults)
                %     p.Results.dim = agent.agent_technique.x_dim;
                end
                
                % if ismethod(obj, 'apply_incremental_step') % Fusion techniques that need an incremental step before the fusion step
                %     obj.apply_incremental_step(varargin{:});
                % end

                result = obj.apply(varargin{:});
                % result = obj.apply('dim', p.Results.dim);
                % Fusion_technique2.update_agent_technique_params(agent, result); % Este update tem que ser dado no agente, não faz sentido ser na técnica de fusão
                agent.fusion_results = result;

            catch exception
                DEBUG(varargin)
                error('Fusion_technique2: Error in social_learning_step - %s', exception.message);
            end
            
        end

        % Utility methods for fusion techniques
        % function normalized_social_matrix = social_matrix_normalize(obj, social_matrix)
        %     % Normalize social matrix for weighted fusion
        %     N = size(social_matrix, 1);
        %     normalized_social_matrix = zeros(N,N);
        %     for a = 1 : N
        %         s = sum(social_matrix(:,a));
        %         if s > 0
        %             normalized_social_matrix(:,a) = social_matrix(:,a) / s;
        %         else
        %             normalized_social_matrix(a,a) = 1; % Self-reliance if no neighbors
        %         end
        %     end
        % end

        % function H_hat_neighbour = get_H_hat_neighbour(obj, collec_H_hat, normalized_social_matrix, N, agent_idx)
        %     % Get weighted H_hat from neighbors for a specific agent
        %     % collec_H_hat: collection of H matrices from all agents
        %     % normalized_social_matrix: normalized adjacency matrix
        %     % N: number of agents
        %     % agent_idx: index of the agent
            
        %     [y_dim, x_dim, ~] = size(collec_H_hat);
        %     H_hat_neighbour = zeros(y_dim, x_dim);
            
        %     for neighbor = 1:N
        %         weight = normalized_social_matrix(neighbor, agent_idx);
        %         if weight > 0
        %             H_hat_neighbour = H_hat_neighbour + weight * collec_H_hat(:,:,neighbor);
        %         end
        %     end
        % end

        % function state_neighbour = get_state_neighbour(obj, collec_states, normalized_social_matrix, N, agent_idx)
        %     % Get weighted state estimates from neighbors for a specific agent
        %     % collec_states: collection of state estimates from all agents
        %     % normalized_social_matrix: normalized adjacency matrix
        %     % N: number of agents
        %     % agent_idx: index of the agent
            
        %     [state_dim, ~] = size(collec_states);
        %     state_neighbour = zeros(state_dim, 1);
            
        %     for neighbor = 1:N
        %         weight = normalized_social_matrix(neighbor, agent_idx);
        %         if weight > 0
        %             state_neighbour = state_neighbour + weight * collec_states(:,neighbor);
        %         end
        %     end
        % end

        % function covariance_neighbour = get_covariance_neighbour(obj, collec_covariances, normalized_social_matrix, N, agent_idx)
        %     % Get weighted covariance matrices from neighbors for a specific agent
        %     % collec_covariances: collection of covariance matrices from all agents
        %     % normalized_social_matrix: normalized adjacency matrix
        %     % N: number of agents
        %     % agent_idx: index of the agent
            
        %     [cov_dim, ~, ~] = size(collec_covariances);
        %     covariance_neighbour = zeros(cov_dim, cov_dim);
            
        %     for neighbor = 1:N
        %         weight = normalized_social_matrix(neighbor, agent_idx);
        %         if weight > 0
        %             covariance_neighbour = covariance_neighbour + weight * collec_covariances(:,:,neighbor);
        %         end
        %     end
        % end

        % % Validation methods
        % function validateInputs(obj, varargin)
        %     % Validate common inputs for fusion techniques
        %     p = inputParser;
            
        %     % Expected inputs for Kalman fusion
        %     addParameter(p, 'collec_H_hat', [], @(x) isnumeric(x) && ndims(x) == 3);
        %     addParameter(p, 'collec_states', [], @(x) isnumeric(x) && ismatrix(x));
        %     addParameter(p, 'collec_covariances', [], @(x) isnumeric(x) && ndims(x) == 3);
        %     addParameter(p, 'social_matrix', [], @(x) isnumeric(x) && ismatrix(x));
            
        %     try
        %         parse(p, varargin{:});
        %     catch exception
        %         error('Fusion_technique2: Invalid input parameters - %s', exception.message);
        %     end
        % end
    end

    methods (Static)
        function update_agent_technique_params(agent)
            
            new_params = agent.fusion_results;

            if isstruct(new_params) && isfield(new_params, 'state_estimate') && isfield(new_params, 'covariance_estimate')
                agent.agent_technique.update_params('state_estimate', new_params.state_estimate, ...
                                                    'covariance_estimate', new_params.covariance_estimate);
            elseif isstruct(new_params) && isfield(new_params, 'state_estimate') && ~isfield(new_params, 'covariance_estimate')
                agent.agent_technique.update_params('state_estimate', new_params.state_estimate);
            end
            agent.xp_hat = new_params.state_estimate;
        end
    end
end
