classdef Single_task < Fusion_technique
    properties
        agent_vec
    end

    methods
        function obj = Single_task(agent_vec)
            obj@Fusion_technique(agent_vec);
        end
        % Assumptions: size(collec_H_hat) = (len(x_hat), len(y), num_of_agents)
        function obj = apply(obj, collec_H_hat, social_matrix)
            % Single Task: weightened H_hats from connected agents 
            n_social_matrix = obj.social_matrix_normalize(social_matrix);
            N = obj.agent_vec.n_agents;
            for a = 1 : N
                H_hat_neighbour = obj.get_H_hat_neighbour(collec_H_hat, n_social_matrix, N, a);
                obj.agent_vec.agents_vec(a).social_learning_step(H_hat_neighbour);
            end
        end
    end
    methods (Static)
        function normalized_social_matrix = social_matrix_normalize(social_matrix)
            N = size(social_matrix, 1);
            normalized_social_matrix = zeros(N,N);
            for a = 1 : N
                s = sum(social_matrix(:,a));
                normalized_social_matrix(:,a) = social_matrix(:,a)/s;
            end
        end
        function H_hat_neighbour = get_H_hat_neighbour(collec_H_hat, n_social_matrix, N, a)
            H_hat_neighbour = zeros(size(collec_H_hat(:,:,1),1), size(collec_H_hat(:,:,1),2));
            for n = 1 : N
                H_hat_neighbour = H_hat_neighbour + n_social_matrix(n,a) * collec_H_hat(:,:,n);
            end
        end
    end
end