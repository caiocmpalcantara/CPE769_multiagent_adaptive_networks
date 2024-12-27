classdef Non_cooperative < Fusion_technique
    properties
        agent_vec
    end

    methods
        function obj = Non_cooperative(agent_vec)
            obj@Fusion_technique(agent_vec);
        end
        % Assumptions: size(collec_x_hat) = (len(x_hat), num_of_agents)
        function obj = apply(obj, collec_H_hat, social_matrix)
            % Do nothing
        end
    end
end