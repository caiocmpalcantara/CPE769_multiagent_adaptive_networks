classdef Consensus_constrain < Fusion_technique
    properties
        agent_vec
    end

    methods
        function obj = Consensus_constrain(agent_vec)
            obj@Fusion_technique(agent_vec);
        end
        % Assumptions: size(collec_H_hat) = (len(x_hat), len(y), num_of_agents)
        function obj = apply(obj, collec_H_hat, social_matrix)
            % Consensus constrain: average from all agents H_hat
            H_hat_all = mean(collec_H_hat, 3);
            for a = 1 : obj.agent_vec.n_agents
                obj.agent_vec.agents_vec(a).social_learning_step(H_hat_all);
            end
        end
    end
end