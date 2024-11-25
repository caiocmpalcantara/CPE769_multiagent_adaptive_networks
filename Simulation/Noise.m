classdef Noise < handle
    properties (Abstract)
        
    end
    methods (Abstract)
       realize(obj, n)
    end
    methods
        function obj = Noise(varargin)
            
        end
    end
end