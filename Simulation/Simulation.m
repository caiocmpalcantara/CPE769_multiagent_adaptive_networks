classdef Simulation < handle
    properties (Abstract)
        
    end
    methods (Abstract)
        simulate(obj)
    end
    methods
        function obj = Simulation(varargin)
            
        end
    end
end