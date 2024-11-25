classdef Fusion_technique < handle
    properties (Abstract)
        
    end
    methods (Abstract)
        %{ apply:
        %  Recebe a referência do vector de agentes e altera suas
        % estimativas com base no critério social.
        %}
        apply(agents_vec, collec_x_hat, social_matrix)  
    end

    methods
        function obj = Fusion_technique(varargin)
            
        end

    end
end