classdef Diff_KF_time_measure < Fusion_technique
    properties
        % agent_vec
        % fusion_strategy     % Strategy for fusion: 'consensus', 'weighted', 'covariance_based'
        % fusion_parameters   % Additional parameters for fusion
        phi_k       % Incremental step state estimate
        P_k         % Incremental step covariance estimate
    end

    methods
        function obj = Diff_KF_time_measure(varargin)
            obj@Fusion_technique(varargin{:});
            
            p = inputParser;
            p.KeepUnmatched = true;

            % Onde definir os pesos? R: Está sendo definido já no Fusion_technique
            
            % default_fusion_strategy = 'weighted';

            % valid_strategies = {'consensus', 'weighted', 'covariance_based'};
            % check_strategy = @(x) any(validatestring(x, valid_strategies));
            % addOptional(p, 'fusion_strategy', default_fusion_strategy, check_strategy);


            % default_fusion_parameters = struct();
            % check_parameters = @(x) isstruct(x);
            % addOptional(p, 'fusion_parameters', default_fusion_parameters, check_parameters);
            
            try
                parse(p, varargin{:});

                obj.phi_k = [];
                obj.P_k = [];
            catch exception
                error('Diff_KF_time_measure: Error in constructor - %s', exception.message);
            end
        end

        function obj = apply_incremental_step(obj, varargin)
            % @brief Apply the incremental step of the Diffusion Kalman Filter with Time Measure Fusion.

            p = inputParser;
            p.KeepUnmatched = true;

            addParameter(p, 'self_agent', [], @(x) isa(x, 'Agent'));
            addParameter(p, 'y_dim', [], @(x) isnumeric(x) && isscalar(x) && (x>0) && (mod(x,1)==0));

            try
                if ~isempty(obj.neighbors)
                    parse(p, varargin{:});

                    DEBUG(sprintf('    Agent %d\n', p.Results.self_agent.getID()))
                    
                    % Incremental step
                    Re = zeros(p.Results.y_dim);
                    phi = p.Results.self_agent.get_posterior_state();
                    DEBUG(phi)
                    P = p.Results.self_agent.get_posterior_covariance('from', 'technique');
                    DEBUG(P)

                    for ind = 1:length(obj.neighbors)
                        if obj.neighbors(ind).getID() ~= p.Results.self_agent.getID()
                            % Re(:,:) = obj.neighbors(ind).agent_technique.R + obj.neighbors(ind).agent_technique.H * obj.neighbors(ind).agent_technique.Pp * obj.neighbors(ind).agent_technique.H';
                            Re(:,:) = obj.neighbors(ind).agent_technique.R + obj.neighbors(ind).agent_technique.H * P * obj.neighbors(ind).agent_technique.H';
                            DEBUG(Re)
                            phi = phi + P * obj.neighbors(ind).agent_technique.H' * pinv(Re) * (obj.neighbors(ind).agent_technique.y - obj.neighbors(ind).agent_technique.H * phi);
                            P = P - P * obj.neighbors(ind).agent_technique.H' * pinv(Re) * obj.neighbors(ind).agent_technique.H * P;
                        end
                    end

                    
                    obj.phi_k = phi;
                    DEBUG(obj.phi_k)
                    obj.P_k = P;
                    DEBUG(obj.P_k)
                end
            catch exception
                if contains(exception.message, 'self_agent')
                    DEBUG(varargin)
                    error('Diff_KF_time_measure: Error in apply_incremental_step - Missing required parameter ''self_agent''.');
                elseif contains(exception.message, 'y_dim')
                    DEBUG(varargin)
                    error('Diff_KF_time_measure: Error in apply_incremental_step - Missing required parameter ''y_dim''.');
                else 
                    % rethrow(exception);
                    error('Diff_KF_time_measure: Error in apply_incremental_step - %s', exception.message);
                end
            end
        end

        function s = apply(obj, varargin)
            % @brief Apply the diffusion step (fusion step) of the Diffusion Kalman Filter with Time Measure Fusion.

            p = inputParser;
            p.KeepUnmatched = true;

            % addParameter(p, 'self_agent', [], @(x) isa(x, 'Agent'));
            addParameter(p, 'dim', [], @(x) isnumeric(x) && isscalar(x) && (x>0) && (mod(x,1)==0));
            % addParameter(p, 'y_dim', [], @(x) isnumeric(x) && isscalar(x) && (x>0) && (mod(x,1)==0));
            
            s = struct();
            try
                if ~isempty(obj.neighbors)
                    parse(p, varargin{:});
                    
                    % Diffusion step (Fusion step)
                    x = zeros(p.Results.dim, 1);
                    for ind = 1:length(obj.neighbors)
                        x(:,1) = x(:,1) + obj.neighbors_weights(ind) * obj.neighbors(ind).fusion_technique.phi_k;
                    end
                    s.state_estimate = x;
                    s.covariance_estimate = obj.P_k;

                    DEBUG(s)

                else
                    error('Diff_KF_time_measure: No neighbors to fuse (even self). It must have at least self.')
                end
                
            catch exception
                if contains(exception.message, 'dim')
                    DEBUG(varargin)
                    error('Diff_KF_time_measure: Error in apply method - Missing required parameter ''dim''.');
                else 
                    % rethrow(exception);
                    error('Diff_KF_time_measure: Error in apply method - %s', exception.message);
                end
            end
        end

    end
end
