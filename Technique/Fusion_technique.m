classdef Fusion_technique < handle
    properties
        neighbors
        neighbors_weights
        % x_hat
    end
    methods (Abstract)
        %{ apply:
        %  Recebe a referência do vector de agentes e altera suas
        % estimativas com base no critério social.
        % Uses varargin for flexible parameter passing compatible with Agent_technique
        %}
        apply(obj, varargin)
        % get_neighbours_estimates(obj)   %TODO: verify if will need varargin
    end

    methods
        function obj = Fusion_technique(varargin)
            p = inputParser;
            p.KeepUnmatched = true;
            
            addParameter(p, 'neighbors', [], @(x) isa(x, 'Agent'));
            addParameter(p, 'neighbors_weights', [], @(x) isnumeric(x));

            try
                parse(p, varargin{:});
                neighbors = p.Results.neighbors;
                neighbors_weights = p.Results.neighbors_weights;

                if ~isempty(neighbors) && ~isempty(neighbors_weights)
                    if length(neighbors) ~= length(neighbors_weights)
                        error('Fusion_technique: Error in constructor - Number of neighbors must match number of weights.');
                    elseif sum(neighbors_weights) > 1+1e-5 && sum(neighbors_weights) < 1-1e-5
                        error('Fusion_technique: Error in constructor - Weights must sum to 1.');
                    end
                end

                obj.neighbors = neighbors;
                obj.neighbors_weights = neighbors_weights;
                % obj.x_hat = [];
            catch exception
                error('Fusion_technique: Error in constructor - %s', exception.message);
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
            
            addParameter(p, 'self_agent', [], @(x) isa(x, 'Agent'));
            addParameter(p, 'dim', [], @(x) isnumeric(x) && isscalar(x) && (x>0) && (mod(x,1)==0));

            try
                parse(p, varargin{:});
                agent = p.Results.self_agent;
                if strcmp('self_agent', p.UsingDefaults)
                    DEBUG(varargin)
                    error('Fusion_technique: Error in social_learning_step - Self agent not provided.')
                % elseif strcmp('dim', p.UsingDefaults)
                %     p.Results.dim = agent.agent_technique.x_dim;
                end

                result = obj.apply(varargin{:});
                % Fusion_technique.update_agent_technique_params(agent, result); % This update does not make sense. The fusion technique should not update the agent technique. The agent should update its technique based on the fusion results.
                agent.fusion_results = result;

            catch exception
                DEBUG(varargin)
                error('Fusion_technique: Error in social_learning_step - %s', exception.message);
            end
            
        end

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
