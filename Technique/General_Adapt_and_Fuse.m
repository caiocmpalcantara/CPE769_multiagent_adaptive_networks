classdef General_Adapt_and_Fuse < Fusion_technique2
    properties
        % agent_vec
        % fusion_strategy     % Strategy for fusion: 'consensus', 'weighted', 'covariance_based'
        % fusion_parameters   % Additional parameters for fusion
        
    end

    methods
        function obj = General_Adapt_and_Fuse(varargin)
            obj@Fusion_technique2(varargin{:});
            
            % p = inputParser;
            % p.KeepUnmatched = true;

            % Onde definir os pesos? R: Está sendo definido já no Fusion_technique
            
            % default_fusion_strategy = 'weighted';

            % valid_strategies = {'consensus', 'weighted', 'covariance_based'};
            % check_strategy = @(x) any(validatestring(x, valid_strategies));
            % addOptional(p, 'fusion_strategy', default_fusion_strategy, check_strategy);


            % default_fusion_parameters = struct();
            % check_parameters = @(x) isstruct(x);
            % addOptional(p, 'fusion_parameters', default_fusion_parameters, check_parameters);
            
            % try
            %     parse(p, varargin{:});

            %     % obj.fusion_strategy = p.Results.fusion_strategy;
            %     % obj.fusion_parameters = p.Results.fusion_parameters;
            % catch exception
            %     error('General_Adapt_and_Fuse: Error in constructor - %s', exception.message);
            % end
        end

        function s = apply(obj, varargin)
            
            p = inputParser;
            p.KeepUnmatched = true;

            % addParameter(p, 'self_agent', [], @(x) isa(x, 'Agent2'));
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

                        % switch class(obj.neighbors(ind).agent_technique)
                        %     % TODO: Do the inv more efficient in the futere
                        %     case 'Kalman2'
                        %         P_inv = P_inv + obj.neighbors_weights(ind) * inv(obj.neighbors(ind).agent_technique.Pp);
                        %     case 'Lms'
                        %     case 'Rls'
                        %     case 'Wiener'
                        %         P_inv = P_inv + obj.neighbors_weights(ind) * inv(obj.neighbors(ind).agent_technique.get_cov_matrix());
                        %     otherwise
                        %         error('General_Adapt_and_Fuse: Unknown agent technique for fusion.')
                        % end
                    end

                    P = pinv(P_inv); 
                    s.covariance_estimate = P;

                    w = zeros(p.Results.dim, 1);
                    for ind = 1:length(obj.neighbors)

                        A = obj.neighbors_weights(ind) * P / obj.neighbors(ind).agent_technique.get_params('covariance_estimate').covariance_estimate;

                        % switch class(obj.neighbors(ind).agent_technique)
                        %     case 'Kalman2'
                        %         % A(:,:,ind) = obj.neighbors_weights(ind) * P / obj.neighbors(ind).agent_technique.Pp; 
                        %         A = obj.neighbors_weights(ind) * P / obj.neighbors(ind).agent_technique.Pp;
                        %     case 'Lms'
                        %     case 'Rls'
                        %     case 'Wiener'
                        %         % A(:,:,ind) = obj.neighbors_weights(ind) * P / obj.neighbors(ind).agent_technique.get_cov_matrix();
                        %         A = obj.neighbors_weights(ind) * P / obj.neighbors(ind).agent_technique.get_cov_matrix(); 
                        %     otherwise
                        %         error('General_Adapt_and_Fuse: Unknown agent technique for fusion.')
                        % end
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

        % function obj = apply_consensus_fusion(obj, collec_H_hat, collec_states, collec_covariances)
        %     % Consensus-based fusion: average all estimates
        %     % Placeholder implementation - to be completed later
            
        %     N = obj.agent_vec.n_agents;
            
        %     if ~isempty(collec_H_hat)
        %         % Average H matrices
        %         H_hat_avg = mean(collec_H_hat, 3);
        %         for a = 1:N
        %             obj.agent_vec.agents_vec(a).social_learning_step(H_hat_avg);
        %         end
        %     end
            
        %     % TODO: Implement state and covariance consensus fusion
        %     % This is a placeholder for future implementation
        % end

        % function obj = apply_weighted_fusion(obj, collec_H_hat, collec_states, collec_covariances, social_matrix)
        %     % Weighted fusion based on network topology
        %     % Placeholder implementation - to be completed later
            
        %     if isempty(social_matrix)
        %         % Fall back to consensus if no social matrix provided
        %         obj = obj.apply_consensus_fusion(collec_H_hat, collec_states, collec_covariances);
        %         return;
        %     end
            
        %     n_social_matrix = obj.social_matrix_normalize(social_matrix);
        %     N = obj.agent_vec.n_agents;
            
        %     if ~isempty(collec_H_hat)
        %         for a = 1:N
        %             H_hat_neighbour = obj.get_H_hat_neighbour(collec_H_hat, n_social_matrix, N, a);
        %             obj.agent_vec.agents_vec(a).social_learning_step(H_hat_neighbour);
        %         end
        %     end
            
        %     % TODO: Implement weighted state and covariance fusion
        %     % This is a placeholder for future implementation
        % end

        % function obj = apply_covariance_fusion(obj, collec_H_hat, collec_states, collec_covariances, social_matrix)
        %     % Covariance-based fusion using information theoretic criteria
        %     % Placeholder implementation - to be completed later
            
        %     % TODO: Implement covariance-based fusion algorithm
        %     % This could include:
        %     % - Information matrix fusion
        %     % - Covariance intersection
        %     % - Optimal linear fusion based on trace minimization
            
        %     % For now, fall back to weighted fusion
        %     obj = obj.apply_weighted_fusion(collec_H_hat, collec_states, collec_covariances, social_matrix);
        % end

        % % Utility methods for future implementation
        % function info_matrix = compute_information_matrix(obj, covariance_matrix)
        %     % Compute information matrix (inverse of covariance)
        %     % Placeholder for future implementation
        %     try
        %         info_matrix = inv(covariance_matrix);
        %     catch
        %         % Handle singular matrices
        %         info_matrix = pinv(covariance_matrix);
        %     end
        % end

        % function fused_covariance = fuse_covariances(obj, covariances, weights)
        %     % Fuse multiple covariance matrices with given weights
        %     % Placeholder for future implementation
        %     [dim, ~, N] = size(covariances);
        %     fused_covariance = zeros(dim, dim);
            
        %     for i = 1:N
        %         fused_covariance = fused_covariance + weights(i) * covariances(:,:,i);
        %     end
        % end
    end
end
