classdef Noon_coop < Fusion_technique
    properties
        
    end

    methods
        function obj = Noon_coop(varargin)
            obj@Fusion_technique(varargin{:});
        end

        function s = apply(obj, varargin)
            % @brief Noon's cooperative filtering technique: Do nothing (dummy)
            p = inputParser;
            p.KeepUnmatched = true;
            
            s = struct();
            try
                DEBUG(s)                
            catch exception
                error('Noon_coop: Error in apply method - %s', exception.message);
            end
        end
    end
end
