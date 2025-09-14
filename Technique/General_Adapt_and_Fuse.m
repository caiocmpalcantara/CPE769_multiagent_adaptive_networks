classdef General_Adapt_and_Fuse < Fusion_technique
    properties
        % agent_vec
        % fusion_strategy     % Strategy for fusion: 'consensus', 'weighted', 'covariance_based'
        % fusion_parameters   % Additional parameters for fusion
        
    end

    methods
        function obj = General_Adapt_and_Fuse(varargin)
            obj@Fusion_technique(varargin{:});
        end

        function s = apply(obj, varargin)
            
            p = inputParser;
            p.KeepUnmatched = true;

            % addParameter(p, 'self_agent', [], @(x) isa(x, 'Agent'));
            addParameter(p, 'dim', [], @(x) isnumeric(x) && isscalar(x) && (x>0) && (mod(x,1)==0));
            
            s = struct();
            try
                if ~isempty(obj.neighbors)
                    parse(p, varargin{:});
                    % self_agent = p.Results.self_agent;
                    
                    P_inv = zeros(p.Results.dim);
                    % A = zeros(p.Results.dim, p.Results.dim, length(obj.neighbors));
                    for ind = 1:length(obj.neighbors)
                        P_inv = P_inv + obj.neighbors_weights(ind) * inv(obj.neighbors(ind).agent_technique.get_params('covariance_estimate').covariance_estimate);
                    end

                    P = pinv(P_inv); 
                    s.covariance_estimate = P;

                    w = zeros(p.Results.dim, 1);
                    for ind = 1:length(obj.neighbors)

                        A = obj.neighbors_weights(ind) * P / obj.neighbors(ind).agent_technique.get_params('covariance_estimate').covariance_estimate;

                        w(:,1) = w(:,1) + A * obj.neighbors(ind).agent_technique.get_params('state_estimate').state_estimate; 
                    end
                    s.state_estimate = w;

                    DEBUG(s)

                else
                    error('General_Adapt_and_Fuse: No neighbors to fuse (even self). It must have at least self.')
                end
                
            catch exception
                if contains(exception.message, 'dim')
                    DEBUG(varargin)
                    error('General_Adapt_and_Fuse: Error in apply method - Missing required parameter ''dim''.');
                else 
                    % rethrow(exception);
                    error('General_Adapt_and_Fuse: Error in apply method - %s', exception.message);
                end
            end
        end

    end
end
