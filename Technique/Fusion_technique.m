classdef Fusion_technique < handle
    properties (Abstract)
        agent_vec
    end
    methods (Abstract)
        %{ apply:
        %  Recebe a referência do vector de agentes e altera suas
        % estimativas com base no critério social.
        %}
        apply(obj, collec_H_hat, social_matrix)
    end

    methods
        function obj = Fusion_technique(agent_vec)
            obj.agent_vec = agent_vec;
        end

    end
end